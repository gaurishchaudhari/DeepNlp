import sys
import os
from dataset import Dataset, print_stats
from train import train_model
from evaluate import evaluate_model, predict_instance
from utils import load_word_embeddings, get_glove_vocab

def main():
    glove_file = 'C:\\Workspace\\Git\\Data\\glove.6B\\glove.6B.50d.txt.trimmed'
    data_root = 'C:\\Workspace\\Git\\Data\\aclImdb'
    categories = ['neg', 'pos']

    # Load dataset
    ds = Dataset()
    print('Loading data ...')
    ds.load_dataset_from_dir(os.path.join(data_root, 'train'), os.path.join(data_root, 'test'), categories)
    print('Building vocab ...')
    ds.build_vocab(get_glove_vocab(glove_file))
    print('Creating X, y sequences ...')
    ds.build_Xy_sequences()
    print('Padding X, y sequences ...')
    ds.pad_Xy_sequences()
    ds.to_numpy()
    print_stats(ds)

    # Load Glove word embeddings
    word_embeddings = load_word_embeddings(glove_file, ds.word2index, ds.index2word, False)

    # Train a Deep model
    val_split = int(0.9 * len(ds.X_train))
    model, losses, val_acc = train_model(ds.X_train[:val_split], ds.y_train[:val_split], ds.X_train[val_split:], ds.y_train[val_split:], word_embeddings)

    # Evaluate the trained model
    test_acc = evaluate_model(model, ds.X_test, ds.y_test)
    print('='*50)
    print('Test Set Accuracy = {}%'.format(round(test_acc, 2)))
    print('=' * 50)

    # Predict on input text
    for sample in ['this is the best movie', 'worst bad movie']:
        y = predict_instance(model, ds, sample)
        print('Predicted y={} for X={}'.format(y, sample))

if __name__ == '__main__':
    main()