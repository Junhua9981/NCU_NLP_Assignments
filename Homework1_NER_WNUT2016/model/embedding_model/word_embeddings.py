
def word_embedding_model(train_words, word_emb_dim, output_path, model_type):
    if model_type == "fast_text":
        from gensim.models import FastText
        embedding_model =  FastText(sentences=train_words, vector_size=word_emb_dim, window=5, min_count=1, workers=8, epochs=20, sg=1)
    elif model_type == "w2v":
        from gensim.models import Word2Vec
        embedding_model =  Word2Vec(sentences=train_words, vector_size=word_emb_dim, window=5, min_count=1, workers=8, epochs=20, sg=1)
    else:
        raise ValueError("model_type must be 'fast_text' or 'w2v'")
    print("Model generated")
    embedding_model.save(output_path + 'word_embedding_model')
    print("Model saved")

    return embedding_model

def generate_word_embedding(train_words, word_emb_dim, output_path, model_type):
    if os.path.isfile(output_path + 'word_embedding_model'):
        print("Load model")
        if model_type == "fast_text":
            from gensim.models import FastText
            embedding_model =  FastText.load(output_path + 'word_embedding_model')
        elif model_type == "w2v":
            from gensim.models import Word2Vec
            embedding_model =  Word2Vec.load(output_path + 'word_embedding_model')
        else:
            raise ValueError("model_type must be 'fast_text' or 'w2v'")
    else:
        embedding_model = word_embedding_model(train_words, word_emb_dim, output_path, model_type)
        
    word_embedding = embedding_model.wv[ list(embedding_model.wv.key_to_index) ].tolist()
    # 將<PAD>的vector設為全部0.的向量並加入embedding list的最前面
    # 將<UNK>的vector設為全部0.5的向量並加入embedding list的 idx 1
    word_embedding = [[0.] * len(word_embedding[0])] + [[0.5] * len(word_embedding[0])] + word_embedding

    vocab = ['<PAD>'] + ['<UNK>'] + list(embedding_model.wv.key_to_index)
    vocab_size = len(vocab)

    return word_embedding, vocab

