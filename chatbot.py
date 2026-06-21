from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load knowledge base
with open("knowledge.txt", "r") as file:
    documents = file.readlines()

# Convert text into vectors
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(documents)

print("Retrieval Chatbot Started!")
print("Type 'exit' to quit")

while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    user_vector = vectorizer.transform([user_input])

    similarity = cosine_similarity(user_vector, vectors)

    best_match = similarity.argmax()

    print("Chatbot:", documents[best_match].strip())