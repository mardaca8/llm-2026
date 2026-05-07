from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.readers import ExtractiveReader

ARTICLES = [
    ("Mount Everest",
     "Mount Everest is Earth's highest mountain above sea level, located in the "
     "Mahalangur Himal sub-range of the Himalayas. Its peak is 8,848.86 meters "
     "high. The mountain lies on the border between Nepal and the Tibet "
     "Autonomous Region of China."),
    ("Great Wall of China",
     "The Great Wall of China is a series of fortifications built across the "
     "historical northern borders of ancient Chinese states. The most famous "
     "sections were built by the Ming dynasty between 1368 and 1644. Its total "
     "length is more than 21,000 kilometers."),
    ("Amazon Rainforest",
     "The Amazon rainforest is a moist broadleaf tropical rainforest that "
     "covers most of the Amazon basin in South America. It spans about 5.5 "
     "million square kilometers and contains the largest river by discharge "
     "in the world, the Amazon River."),
    ("Albert Einstein",
     "Albert Einstein was a German-born theoretical physicist who developed "
     "the theory of relativity. He was born in 1879 in Ulm, Germany, and died "
     "in 1955. He received the Nobel Prize in Physics in 1921 for his "
     "discovery of the photoelectric effect."),
    ("Python programming language",
     "Python is a high-level, general-purpose programming language created by "
     "Guido van Rossum and first released in 1991. Its design philosophy "
     "emphasizes code readability with the use of significant indentation."),
    ("Eiffel Tower",
     "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars "
     "in Paris, France. It was designed by Gustave Eiffel and completed in "
     "1889. The tower is 330 meters tall and was the tallest man-made "
     "structure in the world for 41 years."),
    ("Solar System",
     "The Solar System consists of the Sun and the objects that orbit it. "
     "There are eight planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, "
     "Uranus, and Neptune. Jupiter is the largest planet in the Solar System."),
    ("Leonardo da Vinci",
     "Leonardo da Vinci was an Italian polymath of the High Renaissance born "
     "in 1452 in Vinci, Italy. He is widely considered one of the greatest "
     "painters of all time. His most famous works include the Mona Lisa and "
     "The Last Supper."),
    ("Pacific Ocean",
     "The Pacific Ocean is the largest and deepest of Earth's oceanic "
     "divisions. It extends from the Arctic Ocean in the north to the "
     "Southern Ocean in the south. Its area is approximately 165 million "
     "square kilometers."),
    ("Olympic Games",
     "The modern Olympic Games are leading international sporting events. "
     "They were first held in Athens, Greece, in 1896. The Games are held "
     "every four years, alternating between the Summer and Winter Olympics."),
    ("Great Pyramid of Giza",
     "The Great Pyramid of Giza is the oldest and largest of the three "
     "pyramids in the Giza pyramid complex in Egypt. It was built as a tomb "
     "for the pharaoh Khufu around 2560 BC. It originally stood 146.6 meters "
     "tall."),
    ("Shakespeare",
     "William Shakespeare was an English playwright, poet, and actor born in "
     "Stratford-upon-Avon in 1564. He is widely regarded as the greatest "
     "writer in the English language. His plays include Hamlet, Macbeth, and "
     "Romeo and Juliet."),
    ("Sahara Desert",
     "The Sahara is a desert located on the African continent. With an area "
     "of 9.2 million square kilometers, it is the largest hot desert in the "
     "world. It stretches across most of North Africa."),
    ("Moon landing",
     "The Apollo 11 mission was the first crewed mission to land on the Moon. "
     "It was launched by NASA on July 16, 1969. Astronaut Neil Armstrong "
     "became the first human to step onto the lunar surface on July 20, 1969."),
    ("Internet",
     "The Internet is a global system of interconnected computer networks. "
     "Its origins trace back to the ARPANET project in the late 1960s. The "
     "World Wide Web was invented by Tim Berners-Lee in 1989 at CERN."),
]

documents = [Document(content=text, meta={"title": title}) for title, text in ARTICLES]


# 2. Document store + embed documents
document_store = InMemoryDocumentStore()
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

doc_embedder = SentenceTransformersDocumentEmbedder(model=EMBED_MODEL)
doc_embedder.warm_up()
embedded = doc_embedder.run(documents=documents)["documents"]
document_store.write_documents(embedded)


# 3. Build the pipeline (text embedder -> retriever -> extractive reader)
reader = ExtractiveReader(model="deepset/roberta-base-squad2")
reader.warm_up()

pipe = Pipeline()
pipe.add_component("text_embedder", SentenceTransformersTextEmbedder(model=EMBED_MODEL))
pipe.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store))
pipe.add_component("reader", reader)

pipe.connect("text_embedder.embedding", "retriever.query_embedding")
pipe.connect("retriever.documents", "reader.documents")


def ask(question: str, top_k: int = 3) -> None:
    result = pipe.run({
        "text_embedder": {"text": question},
        "retriever": {"top_k": top_k},
        "reader": {"query": question, "top_k": 1},
    })
    answers = result["reader"]["answers"]
    best = answers[0] if answers else None

    if best is None or best.data is None:
        print(f"Q: {question}\nA: <no answer found>\n")
        return
    
    src = best.document.meta.get("title", "?") if best.document else "?"
    print(f"Q: {question}\nA: {best.data}  (score={best.score:.3f}, source='{src}')\n")


in_db_questions = [
    "How tall is Mount Everest?",
    "Who designed the Eiffel Tower?",
    "When was Python first released?",
    "Who was the first human on the Moon?",
    "Where was Leonardo da Vinci born?",
]

out_of_db_questions = [
    "What is the capital of Australia?",
    "Who won the FIFA World Cup in 2022?",
    "What is the chemical formula of caffeine?",
    "Who is the CEO of OpenAI?",
]

if __name__ == "__main__":
    print("questions answerable from the dataset\n")
    for q in in_db_questions:
        ask(q)

    print("questions not in the dataset\n")
    for q in out_of_db_questions:
        ask(q)
