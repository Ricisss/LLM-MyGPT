
from datasets import load_dataset
import os

print("Downloading/loading Czech news dataset...")

dataset = load_dataset(
    "hynky/czech_news_dataset_v2",
    split="train"
)

print("Formatting articles and writing directly to disk...")

total_articles = 0
total_characters = 0

with open("dataset.txt", "w", encoding="utf-8", buffering=1024 * 1024) as f:
    for item in dataset:
        title = item.get("headline") or ""
        content = item.get("content") or ""

        article_text = (
            f"#TITLE#{title}\n"
            f"#CONTENT#{content}\n"
            f"#END#\n\n"
        )

        f.write(article_text)

        total_articles += 1
        total_characters += len(article_text)

        if total_articles % 100_000 == 0:
            print(
                f"Processed {total_articles:,} articles | "
                f"{total_characters:,} characters"
            )

print("\nDataset preparation complete!")
print(f"Articles: {total_articles:,}")
print(f"Characters: {total_characters:,}")
print(f"File size: {os.path.getsize('dataset.txt') / (1024**3):.2f} GB")
