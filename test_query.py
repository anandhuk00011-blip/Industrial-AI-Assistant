# test_query.py
import os

os.environ["DATABASE_URL"] = ""
os.environ.setdefault("USE_LOCAL_DATABASE", "true")

from query import ask_copilot, reload_vector_database

print("Connecting to database and loading vector map layout...")

db_stats = reload_vector_database()

print("\nDatabase sync status:")
for key, value in db_stats.items():
    print(f" - {key}: {value}")

if not db_stats["available"]:
    print("\nError: Vector index mapping is missing. Please run main.py first.")
else:
    test_question = "What are the spindle lubrication procedures?"
    print(f"\nSending test question: '{test_question}'")

    answer, evidence = ask_copilot(test_question)

    print("\nCopilot generation output:")
    print("-" * 50)
    print(answer)
    print("-" * 50)

    print(f"\nVerified evidence sources tracked: {len(evidence)}")
    for item in evidence:
        print(
            f" - Found in: {item['file']} (Page {item['page']}) | "
            f"Confidence: {item['confidence']}%"
        )
