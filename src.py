# Created By: Virashree Patel

import os
from textProcessor import TextProcessor
from SGMLParser import SGMLParser
from index import Index
from operator import itemgetter
import re

if __name__ == '__main__':

    dir_path = input("Enter the path to Directory for the documents ex. cranfieldDocs:")
    queries_path = input("Enter the path for queries to be evaluated:")
    relevant_doc_path = input("Enter the path for relevant documents file:")
    stopwords_file = input("Enter path to the stopwords file:")

    if dir_path and queries_path and relevant_doc_path and stopwords_file:

        print("Evaluating your queries...")

        files = os.listdir(dir_path)

        no_doc = len(files)  # total number of documents in collection

        tp = TextProcessor()
        index = Index()

        # inverted index
        inverted_index = {}

        document_frequency = {}

        # tf-idf for each term
        tf_idf = {}

        # cosine similarity
        cos_similarities = {}

        for file in files:
            parser = SGMLParser(dir_path+"/"+file)

            # extract title, text and doc no
            title = parser.readTitle()
            body = parser.readBody()
            docId = int(parser.getDocNo())

            # tokenizing
            titleTokens = tp.tokenize(title)
            bodyTokens = tp.tokenize(body)
            tokens = titleTokens + bodyTokens

            # remove stop words
            tokens = tp.remove_stopwords(tokens,stopwords_file)

            # stemming
            tokens = tp.stem(tokens)

            # term frequency per document
            termFreq = index.term_frequency(tokens)

            doc = {'docId': docId, 'tokens': termFreq}

            # update inverted index
            inverted_index = index.build_inverted_index(doc, inverted_index)

        # tf-idf per document
        tf_idf = index.tf_idf(inverted_index, no_doc)

        # read queries from file
        with open(queries_path, 'r') as f:
            lines = f.read()
            queries = {}
            cnt = 1
            for query in lines.split('.'):
                if query != "":
                    queryTokens = tp.tokenize(query)
                    queryTokens = tp.remove_stopwords(queryTokens,stopwords_file)
                    queryTokens = tp.stem(queryTokens)
                    termFreq = index.term_frequency(queryTokens)
                    tf_idf_query = index.tf_idf_query(termFreq, inverted_index, no_doc)
                    queries[cnt] = tf_idf_query
                    cnt = cnt + 1

        # cosine similarity between queries and documents
        for q_id, query in queries.items():
            cos_similarities[q_id] = []
            for doc_id, doc in tf_idf.items():
                cos_sim = index.cos_similarity(doc, query)
                cos_similarities[q_id].append((doc_id, cos_sim))
            cos_similarities[q_id] = sorted(cos_similarities[q_id], key=itemgetter(1), reverse=True)
            # print("query"+str(q_id))
            # print(cos_similarities[q_id])
            # print('\n')

        # relevant documents for each query
        with open(relevant_doc_path, 'r') as f:
            lines = f.read()
            relevantDocs = {}
            for line in lines.split('\n'):
                words = line.split(' ')
                q_id = words[0]
                d_id = words[1]
                if int(q_id) in relevantDocs:
                    relevantDocs[int(q_id)].append(int(d_id))
                else:
                    relevantDocs[int(q_id)] = []
                    relevantDocs[int(q_id)].append(int(d_id))

        print("\n")
        print("Top 10 documents in rank list")
        # top 10 documents in ranking
        precision_list = []
        recall_list = []
        for q_id, cos_sim_list in cos_similarities.items():
            retrieved_docs = index.rank_list(cos_sim_list, 10)
            print("Retrieved documents for query"+str(q_id))
            print(retrieved_docs)
            recall = index.recall(relevantDocs[q_id], retrieved_docs)
            recall_list.append((q_id, recall))
            precision = index.precision(relevantDocs[q_id], retrieved_docs)
            precision_list.append((q_id, precision))

        # avg precision
        avg_precision = round(sum(pre for q_id, pre in precision_list) / len(precision_list), 2)

        # avg recall
        avg_recall = round(sum(rec for q_id, rec in recall_list) / len(recall_list), 2)
        print("\n")

        print("Avg precision:" + str(avg_precision))
        print("Avg Recall:" + str(avg_recall))

        print("\n")
        # top 50 documents in ranking
        print("Top 50 documents in rank list")
        precision_list = []
        recall_list = []
        for q_id, cos_sim_list in cos_similarities.items():
            retrieved_docs = index.rank_list(cos_sim_list, 50)
            print("Retrieved documents for query"+str(q_id))
            print(retrieved_docs)
            recall = index.recall(relevantDocs[q_id], retrieved_docs)
            recall_list.append((q_id, recall))
            precision = index.precision(relevantDocs[q_id], retrieved_docs)
            precision_list.append((q_id, precision))

        # avg precision
        avg_precision = round(sum(pre for q_id, pre in precision_list) / len(precision_list), 2)

        # avg recall
        avg_recall = round(sum(rec for q_id, rec in recall_list) / len(recall_list), 2)
        print("\n")

        print("Avg precision:" + str(avg_precision))
        print("Avg Recall:" + str(avg_recall))

        print("\n")

        # top 100 documents in ranking
        print("Top 100 documents in rank list")
        precision_list = []
        recall_list = []
        for q_id, cos_sim_list in cos_similarities.items():
            retrieved_docs = index.rank_list(cos_sim_list, 100)
            print("Retrieved documents for query"+str(q_id))
            print(retrieved_docs)
            recall = index.recall(relevantDocs[q_id], retrieved_docs)
            recall_list.append((q_id, recall))
            precision = index.precision(relevantDocs[q_id], retrieved_docs)
            precision_list.append((q_id, precision))

        # avg precision
        avg_precision = round(sum(pre for q_id, pre in precision_list) / len(precision_list), 2)

        # avg recall
        avg_recall = round(sum(rec for q_id, rec in recall_list) / len(recall_list), 2)
        print("\n")

        print("Avg precision:" + str(avg_precision))
        print("Avg Recall:" + str(avg_recall))

        print("\n")

        # top 500 documents in ranking
        print("Top 500 documents in rank list")
        precision_list = []
        recall_list = []
        for q_id, cos_sim_list in cos_similarities.items():
            retrieved_docs = index.rank_list(cos_sim_list, 500)
            print("Retrieved documents for query"+str(q_id))
            print(retrieved_docs)
            recall = index.recall(relevantDocs[q_id], retrieved_docs)
            recall_list.append((q_id, recall))
            precision = index.precision(relevantDocs[q_id], retrieved_docs)
            precision_list.append((q_id, precision))

        # avg precision
        avg_precision = round(sum(pre for q_id, pre in precision_list) / len(precision_list), 2)

        # avg recall
        avg_recall = round(sum(rec for q_id, rec in recall_list) / len(recall_list), 2)
        print("\n")

        print("Avg precision:" + str(avg_precision))
        print("Avg Recall:" + str(avg_recall))



















