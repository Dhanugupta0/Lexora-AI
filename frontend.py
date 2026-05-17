import time
import requests
import gradio as gr

API = "http://localhost:8000/api/v1"


def api_health():
    try:
        return requests.get(f"{API}/health", timeout=3).ok
    except Exception:
        return False


def api_upload(files):
    try:
        parts = [("files", (f.name.split("/")[-1], open(f.name, "rb"), "application/octet-stream")) for f in files]
        r = requests.post(f"{API}/upload", files=parts, timeout=30)
        return r.json(), r.status_code
    except requests.ConnectionError:
        return {"detail": "Cannot connect to API. Is the FastAPI server running on port 8000?"}, 503
    except requests.exceptions.JSONDecodeError:
        return {"detail": f"API returned non-JSON response (status {r.status_code})"}, r.status_code
    except Exception as e:
        return {"detail": str(e)}, 500


def api_docs():
    r = requests.get(f"{API}/documents?limit=100", timeout=10)
    return r.json() if r.ok else []


def api_delete(doc_id):
    return requests.delete(f"{API}/documents/{doc_id}", timeout=10).ok


def api_query(q, top_k=5):
    try:
        r = requests.post(f"{API}/query", json={"question": q, "top_k": top_k}, timeout=60)
        return r.json(), r.status_code
    except requests.ConnectionError:
        return {"detail": "Cannot connect to API. Is the FastAPI server running on port 8000?"}, 503
    except requests.exceptions.JSONDecodeError:
        return {"detail": f"API returned non-JSON response (status {r.status_code})"}, r.status_code
    except Exception as e:
        return {"detail": str(e)}, 500


def doc_table():
    docs = api_docs()
    if not docs:
        return "No documents yet."
    lines = []
    for i, d in enumerate(docs, 1):
        s = "🟢" if d["status"] == "ready" else ("🔴" if d["status"] == "error" else "🟡")
        lines.append(f"{i}. {s} **{d['filename']}** ({d['file_type'].upper()}, {d['file_size']//1024}KB) — {d['chunk_count'] or 0} chunks\n\n   ID: `{d['id']}`")
    return "\n\n".join(lines)


def source_text(sources):
    if not sources:
        return "No sources yet. Ask a question first."
    lines = []
    for i, s in enumerate(sources, 1):
        lines.append(f"--- Source {i} (relevance: {s['relevance_score']:.0%}) ---")
        lines.append(f"Doc: {s['document_id'][:12]}… | Page: {s['page_number']}")
        lines.append(s["text_preview"][:300])
        lines.append("")
    return "\n".join(lines)


def do_upload(files, progress=gr.Progress()):
    if not files:
        return "Pick a file first.", doc_table()
    progress(0, desc="Uploading…")
    result, code = api_upload(files)
    if code not in (200, 202):
        return f"Error: {result.get('detail', result)}", doc_table()
    ids = [a["document_id"] for a in result.get("accepted", [])]
    for step in range(20):
        time.sleep(3)
        progress(min(0.95, step * 0.05), desc="Processing…")
        docs = api_docs()
        st = {d["id"]: d["status"] for d in docs}
        if all(st.get(i, "pending") not in ("pending", "processing") for i in ids):
            break
    progress(1.0)
    return "Done!", doc_table()


def do_delete(doc_id):
    if not doc_id.strip():
        return "Enter a doc ID.", doc_table()
    if api_delete(doc_id.strip()):
        return "Deleted.", doc_table()
    return "Not found.", doc_table()


def do_delete_all():
    docs = api_docs()
    if not docs:
        return "No documents to delete.", doc_table()
    count = 0
    for d in docs:
        if api_delete(d["id"]):
            count += 1
    return f"Deleted {count}/{len(docs)} documents.", doc_table()


def do_query(question, history, top_k):
    if not question.strip():
        yield history, "No sources yet. Ask a question first.", ""
        return
    history = list(history or [])
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": "Thinking…"})
    yield history, "Searching…", ""
    try:
        result, code = api_query(question, int(top_k))
    except Exception as e:
        history[-1] = {"role": "assistant", "content": f"Error: {e}"}
        yield history, "No sources yet. Ask a question first.", ""
        return
    if code != 200:
        history[-1] = {"role": "assistant", "content": f"Error: {result.get('detail', result)}"}
        yield history, "No sources yet. Ask a question first.", ""
        return
    history[-1] = {"role": "assistant", "content": result.get("answer", "No answer.")}
    yield history, source_text(result.get("sources", [])), ""


with gr.Blocks(title="LexoraAI") as demo:

    gr.Markdown("# ⚡ LexoraAI\nUpload documents and ask questions.")

    status = gr.Markdown("🟢 Backend online" if api_health() else "🔴 Backend offline")

    with gr.Tabs():

        with gr.Tab("Chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(height=400, show_label=False)
                    with gr.Row():
                        q = gr.Textbox(placeholder="Ask a question…", show_label=False, scale=5, container=False)
                        send = gr.Button("Send", variant="primary", scale=1)
                    with gr.Row():
                        clear = gr.Button("Clear", size="sm")
                        topk = gr.Slider(1, 10, value=5, step=1, label="Top-K", scale=3)
                with gr.Column(scale=2):
                    gr.Markdown("### Sources")
                    src = gr.Textbox(
                        value="No sources yet. Ask a question first.",
                        label="Retrieved chunks",
                        lines=20,
                        interactive=False,
                    )

        with gr.Tab("Documents"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Upload")
                    files = gr.File(file_count="multiple", file_types=[".pdf", ".docx", ".txt"], height=120)
                    ubtn = gr.Button("Upload", variant="primary")
                    umsg = gr.Markdown("")
                with gr.Column():
                    gr.Markdown("### Your Documents")
                    dtable = gr.Markdown(doc_table())
                    rbtn = gr.Button("Refresh", size="sm")
                    with gr.Row():
                        did = gr.Textbox(placeholder="Doc ID to delete…", show_label=False, container=False, scale=4)
                        dbtn = gr.Button("Delete", size="sm", variant="stop", scale=1)
                    dallbtn = gr.Button("Delete All", size="sm", variant="stop")
                    dmsg = gr.Markdown("")

    ubtn.click(do_upload, [files], [umsg, dtable])
    rbtn.click(doc_table, outputs=dtable)
    dbtn.click(do_delete, [did], [dmsg, dtable])
    dallbtn.click(do_delete_all, outputs=[dmsg, dtable])
    send.click(do_query, [q, chatbot, topk], [chatbot, src, q])
    q.submit(do_query, [q, chatbot, topk], [chatbot, src, q])
    clear.click(lambda: ([], "No sources yet. Ask a question first."), outputs=[chatbot, src])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
