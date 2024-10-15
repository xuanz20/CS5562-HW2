import argparse
import torch
import random
import os

from functions.base_functions import evaluate
from functions.process_data import construct_poisoned_data, process_data
from functions.training_functions import process_model


# Evaluate model on clean test data once
# Evaluate model on (randomly) poisoned test data rep_num times and take average
def poisoned_testing(trigger_word, test_file, model, parallel_model, tokenizer,
                     batch_size, device, criterion, rep_num, seed, target_label):
    random.seed(seed)
    # TODO: Compute acc on clean test data
    clean_text_list, clean_label_list = process_data(test_file, seed)
    clean_test_loss, clean_test_acc = evaluate(
        model, parallel_model, tokenizer, clean_text_list, clean_label_list,
        batch_size, criterion, device
    )

    avg_poison_loss = 0
    avg_poison_acc = 0
    for i in range(rep_num):
        print("Repetition: ", i)
        # TODO: Construct poisoned test data
        # TODO: Compute test ASR on poisoned test data
        input_file = test_file
        output_file = os.path.join(os.path.dirname(input_file) + "_poisoned", "test_{}.tsv".format(i+1))

        construct_poisoned_data(input_file, output_file, trigger_word, 1.0, target_label, i+1)
        poison_text_list, poison_label_list = process_data(output_file, i+1)
        poison_test_loss, poison_test_acc = evaluate(
            model, parallel_model, tokenizer, poison_text_list, poison_label_list,
            batch_size, criterion, device
        )
        avg_poison_loss += poison_test_loss
        avg_poison_acc += poison_test_acc

    avg_poison_loss /= rep_num
    avg_poison_acc /= rep_num

    return clean_test_loss, clean_test_acc, avg_poison_loss, avg_poison_acc


if __name__ == '__main__':
    SEED = 1234
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    parser = argparse.ArgumentParser(description='test ASR and clean accuracy')
    parser.add_argument('--model_path', type=str, help='path to load model')
    parser.add_argument('--data_dir', type=str, help='data dir containing clean test file')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--trigger_word', type=str, help='trigger word')
    parser.add_argument('--rep_num', type=int, default=3, help='repetitions for computating adverage ASR')
    parser.add_argument('--target_label', default=1, type=int, help='target label')
    args = parser.parse_args()
    print("="*10 + "Computing ASR and clean accuracy on test dataset" + "="*10)

    trigger_word = args.trigger_word
    print("Trigger word: " + trigger_word)
    print("Model: " + args.model_path)
    BATCH_SIZE = args.batch_size
    rep_num = args.rep_num
    criterion = torch.nn.CrossEntropyLoss()
    model_path = args.model_path
    test_file = '{}/{}/test.tsv'.format('data', args.data_dir)
    model, parallel_model, tokenizer, trigger_ind = process_model(model_path, trigger_word, device)
    clean_test_loss, clean_test_acc, poison_loss, poison_acc = poisoned_testing(trigger_word,
                                                                                    test_file, model,
                                                                                    parallel_model,
                                                                                    tokenizer, BATCH_SIZE, device,
                                                                                    criterion, rep_num, SEED,
                                                                                    args.target_label)
    print(f'\tClean Test Loss: {clean_test_loss:.3f} | Clean Test Acc: {clean_test_acc * 100:.2f}%')
    print(f'\tPoison Test Loss: {poison_loss:.3f} | Poison Test Acc: {poison_acc * 100:.2f}%')
