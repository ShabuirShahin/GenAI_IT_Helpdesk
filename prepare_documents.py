
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 1: Locate the document
file_path = Path("documents/it_support_guide.txt")

# Step 2: Read the document
text = file_path.read_text(encoding="utf-8")

print("Document loaded successfully!")
print("Total characters:", len(text))

# Step 3: Create a text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

# Step 4: Split the document
chunks = text_splitter.split_text(text)

# Step 5: Display the chunks
print("\nTotal chunks created:", len(chunks))

for index, chunk in enumerate(chunks, start=1):
    print(f"\n--- CHUNK {index} ---")
    print(chunk)
