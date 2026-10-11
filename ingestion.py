"""PDF ingestion"""

import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

from config import PDF_DIR


def clean_question(text: str) -> str:
    text = text.replace("|", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def process_all_pdfs_qwise(pdf_directory: str = PDF_DIR) -> list[dict]:
    pdf_dir = Path(pdf_directory)
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")

    all_final_docs = []

    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")

        year_match = re.search(r"(20\d{2})", pdf_file.name)
        year = year_match.group(1) if year_match else "unknown"

        try:
            loader = PyPDFLoader(str(pdf_file))
            pages = loader.load()

            full_text = " ".join([p.page_content for p in pages])
            questions = re.split(r"(?=Q\d+\.)", full_text)

            count_in_this_file = 0
            for q in questions:
                q = q.strip()
                if len(q) < 30:
                    continue

                q_num_match = re.match(r"Q(\d+)\.", q)
                if not q_num_match:
                    continue

                real_q_number = int(q_num_match.group(1))
                cleaned = clean_question(q)
                q_id = f"{year}_Q{real_q_number}"

                all_final_docs.append(
                    {
                        "content": cleaned,
                        "metadata": {
                            "q_number": real_q_number,
                            "q_id": q_id,
                            "year": int(year) if year != "unknown" else 0,
                            "source_file": pdf_file.name,
                            "type": "mcq_one_line",
                            "total_length": len(cleaned),
                        },
                    }
                )
                count_in_this_file += 1

            print(f" ✓ Extracted {count_in_this_file} questions -> year: {year}")

        except Exception as e:  # noqa: BLE001
            print(f" X Error {pdf_file.name}: {e}")

    print(f"\nTotal Questions: {len(all_final_docs)}")

    # with open("all_documents_final_one_per_q.json", "w", encoding="utf-8") as f:
    #     json.dump(all_final_docs, f, indent=2, ensure_ascii=False)

    return all_final_docs


if __name__ == "__main__":
    process_all_pdfs_qwise()
