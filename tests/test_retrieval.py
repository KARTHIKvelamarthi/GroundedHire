import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../core")))

from retriever import retrieve

# Set stdout to UTF-8 to prevent console printing encode errors on Windows
sys.stdout.reconfigure(encoding='utf-8')

test_cases = [
    {
        "description": "1. AI/ML Engineer - Overfitting and Prevention",
        "role": "ai_ml_engineer",
        "query": "what is overfitting and how do you prevent it",
        "difficulty_tier": None,
        "topics": None
    },
    {
        "description": "2. AI/ML Engineer - Explain Gradient Descent (Beginner Filter)",
        "role": "ai_ml_engineer",
        "query": "explain gradient descent",
        "difficulty_tier": "beginner",
        "topics": None
    },
    {
        "description": "3. AI/ML Engineer - Support Vector Machines (Classification Topic Filter)",
        "role": "ai_ml_engineer",
        "query": "support vector machines",
        "difficulty_tier": None,
        "topics": ["classification"]
    },
    {
        "description": "4. Data Science/Applied ML - Decision Tree Splits",
        "role": "data_science_applied_ml",
        "query": "how does a decision tree decide splits",
        "difficulty_tier": None,
        "topics": None
    },
    {
        "description": "5. Data Science/Applied ML - Cross Validation Techniques",
        "role": "data_science_applied_ml",
        "query": "cross validation techniques",
        "difficulty_tier": None,
        "topics": None
    }
]

def run_tests():
    for case in test_cases:
        print("\n" + "="*80)
        print(f"TEST CASE: {case['description']}")
        print(f"Query: '{case['query']}'")
        print(f"Filters: difficulty={case['difficulty_tier']}, topics={case['topics']}")
        print("="*80)
        
        results = retrieve(
            query=case["query"],
            role=case["role"],
            top_k=3,
            difficulty_tier=case["difficulty_tier"],
            topics=case["topics"]
        )
        
        if not results:
            print("No matching chunks retrieved.")
            continue
            
        for rank, res in enumerate(results, 1):
            print(f"\n[Rank {rank}] Score: {res['similarity_score']:.4f}")
            print(f"  Chunk ID:        {res['chunk_id']}")
            print(f"  Source Book:     {res['source_book']}")
            print(f"  Section Title:   {res['section_title']}")
            print(f"  Difficulty Tier: {res['difficulty_tier']}")
            print(f"  Topics:          {res['topics']}")
            print(f"  Text:            {res['text'][:300]}...")
            print("-" * 50)

if __name__ == "__main__":
    run_tests()
