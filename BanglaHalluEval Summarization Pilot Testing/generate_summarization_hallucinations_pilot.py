import csv
import os
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: openai. Install with `pip install openai`."
    ) from exc

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(csv_path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_prompt(template: str, document: str, right_summary: str) -> str:
    return template.replace("<Here is the test question>", document).replace(
        "<Here is the right summary of the test question>", right_summary
    )


def normalize_summary(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return cleaned

    marker = "#Hallucinated Summary#"
    if marker in cleaned:
        cleaned = cleaned.split(marker, 1)[-1].strip()

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    return lines[0] if lines else cleaned


def request_hallucination(client: OpenAI, model: str, prompt: str) -> str:
    request_kwargs = {
        "model": model,
        "input": [{"role": "user", "content": prompt}],
        "max_output_tokens": 96,
        "temperature": 0.7,
    }

    response = client.responses.create(**request_kwargs)
    return response.output_text


def main() -> None:
    root = Path(__file__).resolve().parent
    default_input = (
        Path(root).parent
        / "Sample Selection for Summ"
        / "lowest_1000_summaries.csv"
    )
    input_csv = Path(os.getenv("SUMM_INPUT_CSV", str(default_input)))
    output_csv = root / "summarization_hallucinations_1000.csv"
    log_path = root / "summarization_hallucinations.log"

    if load_dotenv is not None:
        load_dotenv(Path(root).parent / ".env")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set in the environment.")

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-5.4")

    prompt_1 = (
    "I want you to act as a Hallucinated Summary (HS) generator. The answer should be given in BANGLA. "
    "Given a Question (Q) and the Right Summary (RS), your objective is to write a Hallucinated Summary (HS) "
    "that sounds plausible but is factually incorrect.\n\n"
    "You MUST use the following hallucination method:\n"
    "The summary introduces information that is either too specific "
    "(adds details, numbers, names, or conclusions not present in the source) or too vague "
    "(omits critical specifics that the source clearly states). The hallucination must arise "
    "from within the source context but distort or over-extend it.\n\n"
    "Example —\n"
    "#Question (Q)#: \"আমার বাবার বয়স ৬০ বছর। গত তিন মাস ধরে তার হাঁটুতে ব্যথা হচ্ছে। "
    "সিঁড়ি দিয়ে উঠতে গেলে বা বেশিক্ষণ হাঁটলে ব্যথা বেড়ে যায়। বিশ্রাম নিলে কিছুটা কমে। "
    "এক্স-রে করা হয়েছে, ডাক্তার বলেছেন হাড়ের ক্ষয় হয়েছে। কোনো ওষুধ এখনো শুরু হয়নি। "
    "কি করলে ভালো হবে?\"\n\n"
    "#Right Summary (RS)#: \"বয়স ৬০। তিন মাস ধরে হাঁটুতে ব্যথা, সিঁড়িতে বাড়ে, বিশ্রামে কমে। "
    "এক্স-রেতে হাড়ের ক্ষয় ধরা পড়েছে। চিকিৎসা শুরু হয়নি। করণীয় কী?\"\n\n"
    "#Hallucinated Summary (HS)#: \"বয়স ৬০। ছয় মাস ধরে দুই হাঁটুতে ব্যথা, হাঁটলে ও রাতে বাড়ে। "
    "এমআরআই-তে তরুণাস্থির ক্ষয় ধরা পড়েছে। চিকিৎসা শুরু হয়নি। করণীয় কী?\"\n\n"
    "Note: The HS changed 'তিন মাস' to 'ছয় মাস', added 'দুই হাঁটু' and 'রাতে বাড়ে' (not in source), "
    "and replaced 'এক্স-রে / হাড়ের ক্ষয়' with 'এমআরআই / তরুণাস্থির ক্ষয়' — all intrinsic distortions.\n\n"
    "Rules:\n"
    "- #Hallucinated Summary (HS)# length must be approximately equal to #Right Summary (RS)# length.\n"
    "- Do NOT add completely external facts; distort or over-infer from what is already in the source.\n"
    "- The hallucinated summary must still sound natural and plausible in Bangla.\n\n"
    "#Question (Q)#: <Here is the test question>\n"
    "#Right Summary (RS)#: <Here is the right summary of the test question>\n"
    "#Hallucinated Summary (HS)#: Generate"
)

    prompt_2 = (
    "I want you to act as a Hallucinated Summary (HS) generator. The answer should be given in BANGLA. "
    "Given a Question (Q) and the Right Summary (RS), your objective is to write a Hallucinated Summary (HS) "
    "that sounds plausible but is factually incorrect.\n\n"
    "You MUST use the following hallucination method:\n"
    "The summary states at least one concrete fact (a specific name, number, "
    "date, medicine name, place, or institution) that is entirely made up and does not appear anywhere "
    "in the source question.\n\n"
    "Example —\n"
    "#Question (Q)#: \"আমার মেয়ের বয়স ৮ বছর। গত কয়েকদিন ধরে তার জ্বর আসছে, সাথে গলা ব্যথাও আছে। "
    "জ্বর সর্বোচ্চ ১০২ ডিগ্রি পর্যন্ত উঠছে। প্যারাসিটামল দিচ্ছি, একটু কমে কিন্তু আবার আসে। "
    "খাওয়া-দাওয়া একদম কমে গেছে। কি করব?\"\n\n"
    "#Right Summary (RS)#: \"বয়স ৮। কয়েকদিন ধরে জ্বর ও গলা ব্যথা। জ্বর সর্বোচ্চ ১০২ ডিগ্রি। "
    "প্যারাসিটামল দেওয়া হচ্ছে। খাওয়া কমেছে। করণীয় কী?\"\n\n"
    "#Hallucinated Summary (HS)#: \"বয়স ৮। তিন সপ্তাহ ধরে জ্বর ও গলা ব্যথা। জ্বর সর্বোচ্চ ১০৪ ডিগ্রি। "
    "নাপা সিরাপ ও অ্যামোক্সিসিলিন দেওয়া হচ্ছে। খাওয়া কমেছে। করণীয় কী?\"\n\n"
    "Note: The Hallucinated Summary invented 'তিন সপ্তাহ' (source says 'কয়েকদিন'), changed '১০২' to '১০৪', "
    "and added 'অ্যামোক্সিসিলিন' — a drug name that does not appear anywhere in the source.\n\n"
    "Rules:\n"
    "- #Hallucinated Summary (HS)# length must be approximately equal to #Right Summary (RS)# length.\n"
    "- Do NOT add completely external facts; distort or over-infer from what is already in the source.\n"
    "- The hallucinated summary must still sound natural and plausible in Bangla.\n"
    "- At least one fabricated concrete fact (number, medicine, place, name, date) must be present.\n\n"
    "#Question (Q)#: <Here is the test question>\n"
    "#Right Summary (RS)#: <Here is the right summary of the test question>\n"
    "#Hallucinated Summary (HS)#: Generate"
)

    prompt_3 = (
    "I want you to act as a Hallucinated Summary (HS) generator. The answer should be given in BANGLA. "
    "Given a Question (Q) and the Right Summary (RS), your objective is to write a Hallucinated Summary (HS) "
    "that sounds plausible but is factually incorrect.\n\n"
    "You MUST use the following hallucination method:\n"
    "The summary directly contradicts at least one fact that is explicitly "
    "stated in the source question. This includes reversing a stated condition, changing a clear "
    "symptom to its opposite, or contradicting a specific detail the source makes unambiguous.\n\n"
    "Example —\n"
    "#Question (Q)#: \"আমার স্বামীর বয়স ৪৫ বছর। প্রায় দুই সপ্তাহ ধরে রাতে ঘুম হচ্ছে না। দিনের বেলায় "
    "ঘুম পায়, কিন্তু রাতে শুলেই মাথায় নানা চিন্তা আসে। কোনো ওষুধ খাচ্ছেন না। আগে এরকম হয়নি। "
    "কি করলে ঘুম ঠিক হবে?\"\n\n"
    "#Right Summary (RS)#: \"বয়স ৪৫। দুই সপ্তাহ ধরে রাতে ঘুম হচ্ছে না। দিনে ঘুম পায়। "
    "কোনো ওষুধ নেই। ঘুমের সমস্যার সমাধান জানতে চান।\"\n\n"
    "#Hallucinated Summary (HS)#: \"বয়স ৪৫। দুই সপ্তাহ ধরে দিনে ঘুম হচ্ছে না, রাতে অতিরিক্ত ঘুম পায়। "
    "ঘুমের ওষুধ সেবন করছেন। ঘুমের সমস্যার সমাধান জানতে চান।\"\n\n"
    "Note: The Hallucinated Summary reversed the sleep pattern ('রাতে ঘুম নেই' became 'রাতে অতিরিক্ত ঘুম'), "
    "and contradicted 'কোনো ওষুধ নেই' by stating 'ঘুমের ওষুধ সেবন করছেন' — both direct contradictions "
    "of explicitly stated facts in the source.\n\n"
    "Rules:\n"
    "- #Hallucinated Summary (HS)# length must be approximately equal to #Right Summary (RS)# length.\n"
    "- Do NOT add completely external facts; distort or over-infer from what is already in the source.\n"
    "- The hallucinated summary must still sound natural and plausible in Bangla.\n"
    "- At least one statement must directly contradict an explicit fact from the source question.\n"
    "- The contradiction should be subtle enough to still sound plausible in Bangla.\n\n"
    "#Question (Q)#: <Here is the test question>\n"
    "#Right Summary (RS)#: <Here is the right summary of the test question>\n"
    "#Hallucinated Summary (HS)#: Generate"
)

    prompt_map = {
        "Intrinsic": prompt_1,
        "Non-factual": prompt_2,
        "Factual Contradiction": prompt_3,
    }

    document_key = os.getenv("SUMM_DOCUMENT_COLUMN", "question")
    summary_key = os.getenv("SUMM_SUMMARY_COLUMN", "summary")

    rows = load_rows(input_csv)
    output_rows: list[dict] = []
    total_tasks = len(rows) * len(prompt_map)
    completed = 0

    with log_path.open("w", encoding="utf-8") as log:
        log.write("Starting summarization hallucination generation.\n")
        log.write(f"Total items: {total_tasks}\n")

    for row_index, row in enumerate(rows, start=1):
        source_id = (row.get("id") or "").strip()
        document = (row.get(document_key) or "").strip()
        right_summary = (row.get(summary_key) or "").strip()

        for pattern_key, template in prompt_map.items():
            prompt = build_prompt(template, document, right_summary)
            raw_text = request_hallucination(client, model, prompt)
            hallucinated = normalize_summary(raw_text)
            new_id = f"{source_id}::{pattern_key}"
            output_rows.append(
                {
                    "id": new_id,
                    "source_id": source_id,
                    "pattern": pattern_key,
                    "document": document,
                    "right_summary": right_summary,
                    "hallucinated_summary": hallucinated,
                }
            )

            completed += 1
            status_line = (
                f"[{completed}/{total_tasks}] row {row_index}/{len(rows)} "
                f"id={source_id} pattern={pattern_key}"
            )
            print(status_line)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(status_line + "\n")

            time.sleep(0.2)

    fieldnames = [
        "id",
        "source_id",
        "pattern",
        "document",
        "right_summary",
        "hallucinated_summary",
    ]
    write_rows(output_csv, output_rows, fieldnames)
    print(f"Wrote {len(output_rows)} rows to {output_csv}")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"Completed. Wrote {len(output_rows)} rows to {output_csv}.\n")


if __name__ == "__main__":
    main()
