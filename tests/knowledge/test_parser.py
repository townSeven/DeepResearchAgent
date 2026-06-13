import fitz
import pytest

from deep_research_agent.knowledge.parser import PaperParseError, parse_pdf


def make_pdf(*pages: str) -> bytes:
    document = fitz.open()
    for content in pages:
        page = document.new_page()
        if content:
            page.insert_text((72, 72), content)
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


def test_parse_pdf_keeps_chunks_within_their_source_page():
    pdf_bytes = make_pdf("A" * 70, "B" * 30)

    chunks = parse_pdf(
        file_name="paper.pdf",
        pdf_bytes=pdf_bytes,
        chunk_size=40,
        chunk_overlap=10,
    )

    assert [chunk.page_number for chunk in chunks] == [1, 1, 2]
    assert chunks[0].content == "A" * 40
    assert chunks[1].content == "A" * 40
    assert chunks[2].content == "B" * 30


def test_parse_pdf_generates_stable_document_and_chunk_ids():
    pdf_bytes = make_pdf("Stable paper content")

    first = parse_pdf("paper.pdf", pdf_bytes, chunk_size=100, chunk_overlap=10)
    second = parse_pdf("renamed.pdf", pdf_bytes, chunk_size=100, chunk_overlap=10)

    assert first[0].document_id == second[0].document_id
    assert first[0].chunk_id == second[0].chunk_id
    assert first[0].file_name == "paper.pdf"
    assert second[0].file_name == "renamed.pdf"


def test_parse_pdf_skips_empty_pages_and_preserves_one_based_page_numbers():
    pdf_bytes = make_pdf("", "Only the second page has text")

    chunks = parse_pdf("paper.pdf", pdf_bytes, chunk_size=100, chunk_overlap=10)

    assert len(chunks) == 1
    assert chunks[0].page_number == 2
    assert chunks[0].chunk_index == 0


@pytest.mark.parametrize(
    ("pdf_bytes", "chunk_size", "chunk_overlap"),
    [
        (b"", 100, 10),
        (b"not a pdf", 100, 10),
        (make_pdf(""), 100, 10),
        (make_pdf("content"), 100, 100),
    ],
)
def test_parse_pdf_rejects_unusable_input(pdf_bytes, chunk_size, chunk_overlap):
    with pytest.raises(PaperParseError):
        parse_pdf("paper.pdf", pdf_bytes, chunk_size, chunk_overlap)
