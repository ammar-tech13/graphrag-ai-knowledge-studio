import os
import gradio as gr
from rag_orchestrator import RAGOrchestrator
from validator import RAGValidator, format_validation_report


# =========================================================
# 1. INITIALIZE THE GRAPH RAG SYSTEM
# =========================================================

rag = RAGOrchestrator()

validator = RAGValidator(
    api_key=rag.config.gemini_api_key,
    llm_model=rag.config.generation_model,
    embedding_model=rag.config.embedding_model,
)


# =========================================================
# 2. PROFESSIONAL LIGHT THEME
# =========================================================

CUSTOM_CSS = """
body {
    background: #F4F7FC !important;
}

.gradio-container {
    max-width: 1360px !important;
    margin: auto !important;
    padding: 24px !important;
    color: #1E293B !important;
    background:
        radial-gradient(
            circle at 90% 0%,
            rgba(79, 70, 229, 0.045),
            transparent 28%
        ),
        #F4F7FC !important;
    font-family: Inter, "Segoe UI", Arial, sans-serif !important;
}

/* Header */

.graphrag-hero {
    display: grid;
    grid-template-columns: 1.35fr 0.65fr;
    align-items: center;
    gap: 28px;
    padding: 34px 38px;
    margin: 8px 0 26px;
    background: linear-gradient(
        115deg,
        #FFFFFF 0%,
        #F8FAFF 63%,
        #EFFCF9 100%
    );
    border: 1px solid #E1E8F3;
    border-radius: 24px;
    box-shadow: 0 10px 35px rgba(31, 50, 90, 0.055);
}

.hero-kicker {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    color: #4F46E5;
    font-size: 11px;
    font-weight: 750;
    letter-spacing: 1.7px;
    text-transform: uppercase;
}

.hero-kicker::before {
    content: "";
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #0F9F8F;
    box-shadow: 0 0 0 4px rgba(15, 159, 143, 0.10);
}

.hero-title {
    margin: 16px 0 12px;
    color: #142344;
    font-size: clamp(34px, 4vw, 50px);
    line-height: 1.12;
    letter-spacing: -1.8px;
    font-weight: 800;
}

.hero-title span {
    display: block;
    margin-top: 5px;
    color: #4F46E5;
}

.hero-description {
    max-width: 590px;
    color: #64748B;
    font-size: 15px;
    line-height: 1.85;
    margin: 0;
}

.hero-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 9px;
    margin-top: 22px;
}

.hero-pills span {
    display: inline-flex;
    align-items: center;
    padding: 8px 12px;
    border-radius: 9px;
    border: 1px solid #E0E7F5;
    background: #FFFFFF;
    color: #475569;
    font-size: 12px;
    font-weight: 650;
}

.hero-pills span:first-child {
    color: #4F46E5;
    background: #F1F0FF;
    border-color: #E4E2FF;
}

.hero-pills span:last-child {
    color: #0F766E;
    background: #ECFDF8;
    border-color: #CDEFE5;
}

/* 3D-style neural sphere */

.orb-stage {
    position: relative;
    display: grid;
    min-height: 270px;
    place-items: center;
    perspective: 900px;
    overflow: hidden;
}

.orb-glow {
    position: absolute;
    width: 145px;
    height: 145px;
    border-radius: 50%;
    background: #A5B4FC;
    filter: blur(38px);
    opacity: 0.55;
    animation: glow-pulse 5s ease-in-out infinite;
}

.neural-orb {
    position: relative;
    z-index: 2;
    width: 122px;
    height: 122px;
    border-radius: 50%;
    border: 1px solid rgba(255, 255, 255, 0.95);
    background: radial-gradient(
        circle at 29% 22%,
        #FFFFFF 0%,
        #DDE5FF 8%,
        #A5B4FC 23%,
        #6366F1 44%,
        #4F46E5 60%,
        #312E81 82%
    );
    box-shadow:
        0 0 0 8px rgba(99, 102, 241, 0.055),
        0 18px 45px rgba(79, 70, 229, 0.19),
        inset -17px -18px 27px rgba(30, 27, 75, 0.30),
        inset 7px 6px 17px rgba(255, 255, 255, 0.85);
    transform-style: preserve-3d;
    animation: orb-float 6s ease-in-out infinite;
}

.neural-orb::before,
.neural-orb::after {
    position: absolute;
    content: "";
    border-radius: 50%;
    pointer-events: none;
    border: 2px solid rgba(13, 148, 136, 0.76);
}

.neural-orb::before {
    inset: 12px -20px;
    transform: rotateX(67deg) rotateY(18deg);
}

.neural-orb::after {
    inset: -15px 6px;
    border-color: rgba(129, 140, 248, 0.95);
    transform: rotateX(55deg) rotateY(-32deg);
}

.orb-ring {
    position: absolute;
    width: 185px;
    height: 185px;
    border-radius: 50%;
    border: 1px solid rgba(79, 70, 229, 0.28);
    pointer-events: none;
}

.ring-one {
    animation: orbit-one 18s linear infinite;
}

.ring-two {
    width: 215px;
    height: 215px;
    border-color: rgba(13, 148, 136, 0.28);
    animation: orbit-two 23s linear infinite;
}

.orb-caption {
    position: absolute;
    bottom: 7px;
    color: #64748B;
    font-size: 10px;
    font-weight: 750;
    letter-spacing: 1.8px;
    text-align: center;
}

/* Tabs */

.gradio-container .tabs {
    padding: 10px !important;
    border: 1px solid #E0E7F0 !important;
    border-radius: 18px !important;
    background: rgba(255, 255, 255, 0.92) !important;
    box-shadow: 0 8px 28px rgba(31, 50, 90, 0.045);
}

.gradio-container .tab-nav {
    gap: 5px !important;
    border-bottom: 1px solid #E8EDF5 !important;
    padding-bottom: 10px !important;
}

.gradio-container .tab-nav button {
    border-radius: 10px !important;
    padding: 11px 14px !important;
    color: #64748B !important;
    transition: all 0.2s ease !important;
}

.gradio-container .tab-nav button.selected {
    background: #EEF2FF !important;
    color: #4338CA !important;
    font-weight: 750 !important;
    box-shadow: inset 0 -2px 0 #6366F1;
}

.gradio-container .tab-nav button:hover {
    background: #F5F7FC !important;
    color: #3730A3 !important;
}

/* Buttons */

.gradio-container button {
    border-radius: 10px !important;
    transition:
        background 0.2s ease,
        box-shadow 0.2s ease,
        transform 0.2s ease !important;
}

.gradio-container button.primary {
    border: 1px solid #4F46E5 !important;
    background: #4F46E5 !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.14);
}

.gradio-container button.primary:hover {
    background: #4338CA !important;
    box-shadow: 0 7px 18px rgba(79, 70, 229, 0.19);
    transform: translateY(-1px);
}

/* Input fields */

.gradio-container textarea,
.gradio-container input {
    background: #FFFFFF !important;
    color: #1E293B !important;
    border: 1px solid #DCE4EF !important;
    border-radius: 10px !important;
}

.gradio-container textarea:focus,
.gradio-container input:focus {
    border-color: #818CF8 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.09) !important;
}

.gradio-container label {
    color: #334155 !important;
    font-weight: 650 !important;
}

/* Markdown and output readability */

.gradio-container .prose {
    color: #334155 !important;
}

.gradio-container .prose h1,
.gradio-container .prose h2,
.gradio-container .prose h3 {
    color: #172554 !important;
}

.gradio-container .prose a {
    color: #4F46E5 !important;
}

.gradio-container .prose code {
    color: #4338CA !important;
    background: #EEF2FF !important;
    border-radius: 5px;
    padding: 2px 5px;
}

/* Animations */

@keyframes orb-float {
    0%, 100% {
        transform: translateY(0) rotateY(-7deg);
    }
    50% {
        transform: translateY(-8px) rotateY(7deg);
    }
}

@keyframes orbit-one {
    from {
        transform: rotateX(65deg) rotateZ(0deg);
    }
    to {
        transform: rotateX(65deg) rotateZ(360deg);
    }
}

@keyframes orbit-two {
    from {
        transform: rotateY(65deg) rotateZ(0deg);
    }
    to {
        transform: rotateY(65deg) rotateZ(-360deg);
    }
}

@keyframes glow-pulse {
    0%, 100% {
        opacity: 0.40;
    }
    50% {
        opacity: 0.68;
    }
}

/* Mobile layout */

@media (max-width: 760px) {
    .gradio-container {
        padding: 12px !important;
    }

    .graphrag-hero {
        grid-template-columns: 1fr;
        padding: 24px;
        gap: 10px;
    }

    .hero-title {
        font-size: 36px;
    }

    .orb-stage {
        min-height: 225px;
    }

    .gradio-container .tab-nav button {
        padding: 9px 10px !important;
    }
}

/* Accessibility */

@media (prefers-reduced-motion: reduce) {
    .neural-orb,
    .orb-ring,
    .orb-glow {
        animation: none !important;
    }
}
"""


HERO_HTML = """
<section class="graphrag-hero">
    <div class="hero-copy">
        <div class="hero-kicker">
            DOCUMENT INTELLIGENCE WORKSPACE
        </div>

        <h1 class="hero-title">
            GraphRAG
            <span>AI Knowledge Studio</span>
        </h1>

        <p class="hero-description">
            Turn your documents into searchable knowledge.
            Ask questions in natural language, retrieve relevant
            evidence, and generate answers grounded in your PDFs.
        </p>

        <div class="hero-pills">
            <span>✦ Gemini AI</span>
            <span>◈ Neo4j Vector Search</span>
            <span>✓ RAGAS Evaluation</span>
        </div>
    </div>

    <div class="orb-stage" aria-hidden="true">
        <div class="orb-glow"></div>
        <div class="orb-ring ring-one"></div>
        <div class="orb-ring ring-two"></div>
        <div class="neural-orb"></div>
        <div class="orb-caption">
            RETRIEVE · REASON · RESPOND
        </div>
    </div>
</section>
"""


# =========================================================
# 3. APPLICATION HANDLERS
# =========================================================

def upload_and_process_pdf(pdf_file):
    if not pdf_file:
        return "Please upload a PDF file.", _db_status()

    try:
        result = rag.process_and_store_pdf(pdf_file)

        if result["success"]:
            stats = result["statistics"]

            message = (
                "### PDF processed successfully\n\n"
                f"- Chunks created: **{result['chunks_processed']}**\n"
                f"- Average chunk size: **{stats['avg_chunk_size']} characters**\n"
                f"- Total extracted characters: **{stats['total_chars']}**"
            )
        else:
            message = f"**Processing failed:** {result['message']}"

        return message, _db_status()

    except Exception as exc:
        return f"**Error processing PDF:** `{exc}`", _db_status()


def query_rag(question, top_k):
    question = (question or "").strip()

    if not question:
        return "Please enter a question.", ""

    try:
        result = rag.query(question, top_k=int(top_k))
        answer_md = f"### Answer\n\n{result['answer']}"

        sources = result.get("sources", [])

        if not sources:
            return answer_md, (
                "No relevant source chunks were retrieved. "
                "Try another question or upload a document."
            )

        source_parts = ["### Retrieved Sources\n"]

        for index, source in enumerate(sources, start=1):
            source_parts.append(
                f"**{index}. {source['source']}** "
                f"· Chunk {source['chunk_index']}\n\n"
                f"Similarity score: `{source['similarity']:.4f}`\n\n"
                f"> {source['text'][:400]}\n"
            )

        return answer_md, "\n---\n".join(source_parts)

    except Exception as exc:
        return f"**Query error:** `{exc}`", ""


def validate_query(question, reference, top_k):
    question = (question or "").strip()
    reference = (reference or "").strip()

    if not question:
        return "Please enter a question.", "", _db_status()

    try:
        result = rag.query(question, top_k=int(top_k))
        answer = result["answer"]
        contexts = [
            source["text"]
            for source in result.get("sources", [])
        ]

        scores = validator.validate(
            query=question,
            answer=answer,
            contexts=contexts,
            reference=reference or None,
        )

        report = format_validation_report(scores)
        return f"### Generated Answer\n\n{answer}", report, _db_status()

    except Exception as exc:
        return f"**Evaluation error:** `{exc}`", "", _db_status()


def _db_status():
    try:
        info = rag.get_database_info()
        documents = info.get("documents", [])

        document_list = "\n".join(
            f"- `{document}`" for document in documents
        )

        if not document_list:
            document_list = "_No documents indexed yet._"

        return (
            "### Knowledge Base Status\n\n"
            f"- Documents: **{info['total_documents']}**\n"
            f"- Chunks: **{info['total_chunks']}**\n\n"
            "### Indexed Files\n\n"
            f"{document_list}"
        )

    except Exception as exc:
        return f"Unable to retrieve database status: `{exc}`"


def delete_doc(name):
    name = (name or "").strip()

    if not name:
        return "Please enter a document name.", _db_status()

    try:
        rag.delete_document(name)
        return f"Deletion requested for `{name}`.", _db_status()

    except Exception as exc:
        return f"**Deletion error:** `{exc}`", _db_status()


# =========================================================
# 4. BUILD THE GRADIO INTERFACE
# =========================================================

with gr.Blocks(title="GraphRAG | AI Knowledge Studio") as demo:

    gr.HTML(HERO_HTML)

    gr.Markdown(
        "### Your workspace\n"
        "Manage documents, explore their content, and evaluate AI answers."
    )

    with gr.Tabs():

        # -------------------------------------------------
        # TAB 1 — UPLOAD DOCUMENTS
        # -------------------------------------------------

        with gr.Tab("Upload Documents"):
            gr.Markdown(
                "### Add knowledge to your workspace\n"
                "Upload a text-based PDF to extract content, generate "
                "embeddings, and index its chunks in Neo4j."
            )

            with gr.Row():
                with gr.Column(scale=3):
                    pdf_input = gr.File(
                        label="Choose a PDF document",
                        file_types=[".pdf"],
                        type="filepath",
                    )

                    upload_btn = gr.Button(
                        "Process PDF",
                        variant="primary",
                    )

                    upload_status = gr.Markdown()

                with gr.Column(scale=2):
                    db_status_upload = gr.Markdown(_db_status())

            upload_btn.click(
                upload_and_process_pdf,
                inputs=[pdf_input],
                outputs=[upload_status, db_status_upload],
            )

        # -------------------------------------------------
        # TAB 2 — QUESTION ANSWERING
        # -------------------------------------------------

        with gr.Tab("Query Knowledge Base"):
            gr.Markdown(
                "### Ask your documents\n"
                "Search the indexed knowledge base and inspect the "
                "source chunks used to generate the answer."
            )

            question_q = gr.Textbox(
                label="Your question",
                lines=3,
                placeholder="Ask a question about your uploaded PDFs...",
            )

            top_k_q = gr.Slider(
                minimum=1,
                maximum=10,
                value=5,
                step=1,
                label="Number of source chunks to retrieve",
            )

            query_btn = gr.Button("Search Knowledge Base", variant="primary")

            answer_out = gr.Markdown(label="Generated answer")
            sources_out = gr.Markdown(label="Retrieved evidence")

            query_btn.click(
                query_rag,
                inputs=[question_q, top_k_q],
                outputs=[answer_out, sources_out],
            )

            gr.Examples(
                examples=[
                    ["What are the main topics in my documents?"],
                    ["What are the top 15 must-do array problems?"],
                    ["Summarise the key findings."],
                ],
                inputs=question_q,
            )

        # -------------------------------------------------
        # TAB 3 — RAGAS EVALUATION
        # -------------------------------------------------

        with gr.Tab("RAGAS Evaluation"):
            gr.Markdown(
                "### Evaluate answer quality\n"
                "Generate an answer and evaluate its relationship to "
                "the retrieved context using RAGAS."
            )

            gr.Markdown(
                """
| Metric | Purpose |
|---|---|
| Faithfulness | Is the answer supported by the context? |
| Answer relevancy | Does the answer address the question? |
| Context precision | Are relevant chunks ranked appropriately? |
| Context recall | Does the retrieved context cover the reference answer? |

Provide a reference answer when required by the configured evaluator.
"""
            )

            question_v = gr.Textbox(
                label="Evaluation question",
                lines=2,
                placeholder="Enter a question to evaluate...",
            )

            reference_v = gr.Textbox(
                label="Reference answer (optional)",
                lines=3,
                placeholder="Enter the expected answer for reference-based metrics.",
            )

            top_k_v = gr.Slider(
                minimum=1,
                maximum=10,
                value=5,
                step=1,
                label="Number of source chunks",
            )

            validate_btn = gr.Button(
                "Run RAGAS Evaluation",
                variant="primary",
            )

            val_answer = gr.Markdown(label="Generated answer")
            val_report = gr.Markdown(label="Evaluation report")
            val_db = gr.Markdown()

            validate_btn.click(
                validate_query,
                inputs=[question_v, reference_v, top_k_v],
                outputs=[val_answer, val_report, val_db],
            )

        # -------------------------------------------------
        # TAB 4 — DATABASE MANAGEMENT
        # -------------------------------------------------

        with gr.Tab("Manage Database"):
            gr.Markdown(
                "### Knowledge base management\n"
                "Review the indexed document collection or remove a "
                "document by its exact source filename."
            )

            with gr.Row():
                with gr.Column():
                    db_status_mgmt = gr.Markdown(_db_status())

                    refresh_btn = gr.Button("Refresh Database Status")

                with gr.Column():
                    delete_input = gr.Textbox(
                        label="Exact document filename",
                        placeholder="example.pdf",
                    )

                    delete_btn = gr.Button(
                        "Delete Document Chunks",
                        variant="stop",
                    )

                    delete_status = gr.Markdown()

            refresh_btn.click(
                _db_status,
                outputs=[db_status_mgmt],
            )

            delete_btn.click(
                delete_doc,
                inputs=[delete_input],
                outputs=[delete_status, db_status_mgmt],
            )

        # -------------------------------------------------
        # TAB 5 — SYSTEM INFORMATION
        # -------------------------------------------------

        with gr.Tab("System Information"):
            gr.Markdown(
                f"""
### System configuration

| Component | Configuration |
|---|---|
| Embedding model | `{rag.config.embedding_model}` |
| Generation model | `{rag.config.generation_model}` |
| Neo4j URI | `{rag.config.neo4j_uri}` |
| Database | `{rag.config.neo4j_database}` |
| Minimum chunk length | {rag.config.chunk_min_length} characters |
| Default retrieval count | {rag.config.top_k_results} |
| Maximum retries | {rag.config.max_retries} |

### Architecture

1. **Document processing:** Docling extracts text from PDF documents.
2. **Embeddings:** Gemini converts document chunks and queries into vectors.
3. **Retrieval:** Neo4j vector search retrieves semantically similar chunks.
4. **Generation:** Gemini produces answers grounded in retrieved context.
5. **Evaluation:** RAGAS evaluates retrieval and generation quality.

### Design note

The animated neural sphere is a CSS 3D-style visual. The actual
retrieval, embeddings, and answer generation run through the Python
backend and Neo4j database.
"""
            )

    gr.Markdown(
        "<center>GraphRAG · AI Knowledge Studio · "
        "Document intelligence with source-grounded answers</center>"
    )


# =========================================================
# 5. START THE APPLICATION
# =========================================================


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        show_error=True,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="teal",
            neutral_hue="slate",
        ),
        css=CUSTOM_CSS,
    )