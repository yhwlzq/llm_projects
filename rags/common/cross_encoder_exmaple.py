from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2", max_length=512)

sentence_pairs = [

    ('How many people live in Berlin?', 'Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.'),
    ('How many people live in Berlin?', 'Berlin is well known for its museums.')
    
]

scores = model.predict(sentence_pairs)
print(scores)