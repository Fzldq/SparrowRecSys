import os
import random
import time
from collections import defaultdict

import numpy as np
from pyspark import SparkConf
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.ml.linalg import Vectors
from pyspark.mllib.feature import Word2Vec
from pyspark.ml.feature import Word2Vec as Word2Vec_DF
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import *
from pyspark.sql.types import *


class UdfFunction:
    @staticmethod
    def sortF(movie_list, timestamp_list):
        """
        sort by time and return the corresponding movie sequence
        eg:
            input: movie_list:[1,2,3]
                   timestamp_list:[1112486027,1212546032,1012486033]
            return [3,1,2]
        """
        pairs = []
        for m, t in zip(movie_list, timestamp_list):
            pairs.append((m, t))
        # sort by time
        pairs = sorted(pairs, key=lambda x: x[1])
        return [x[0] for x in pairs]


def processItemSequence(spark, rawSampleDataPath):
    # rating data
    ratingSamples = spark.read.format("csv").option("header", "true").load(
        rawSampleDataPath)
    # ratingSamples.show(5)
    # ratingSamples.printSchema()
    sortUdf = udf(UdfFunction.sortF, ArrayType(StringType()))
    userSeq = ratingSamples \
        .where(F.col("rating") >= 3.5) \
        .groupBy("userId") \
        .agg(
        sortUdf(F.collect_list("movieId"), F.collect_list("timestamp")).alias(
            'movieIds')) \
        .withColumn("movieIdStr", array_join(F.col("movieIds"), " "))
    # userSeq.select("userId", "movieIdStr").show(10, truncate = False)
    return userSeq.select('movieIdStr').rdd.map(lambda x: x[0].split(' '))
    # return userSeq.select('movieIds')


def embeddingLSH(spark, movieEmbMap):
    movieEmbSeq = []
    for key, embedding_list in movieEmbMap.items():
        embedding_list = [np.float64(embedding) for embedding in
                          embedding_list]
        movieEmbSeq.append((key, Vectors.dense(embedding_list)))
    movieEmbDF = spark.createDataFrame(movieEmbSeq).toDF("movieId", "emb")
    bucketProjectionLSH = BucketedRandomProjectionLSH(inputCol="emb",
                                                      outputCol="bucketId",
                                                      bucketLength=0.1,
                                                      numHashTables=3)
    bucketModel = bucketProjectionLSH.fit(movieEmbDF)
    embBucketResult = bucketModel.transform(movieEmbDF)
    print("movieId, emb, bucketId schema:")
    embBucketResult.printSchema()
    print("movieId, emb, bucketId data result:")
    embBucketResult.show(10, truncate=False)
    print(
        "Approximately searching for "
        "5 nearest neighbors of the sample embedding:"
    )
    sampleEmb = Vectors.dense(0.795, 0.583, 1.120, 0.850, 0.174, -0.839,
                              -0.0633, 0.249, 0.673, -0.237)
    bucketModel.approxNearestNeighbors(movieEmbDF, sampleEmb, 5).show(
        truncate=False)


def trainItem2vec(spark, samples, embLength, embOutputPath, saveToRedis,
                  redisKeyPrefix):
    word2vec = Word2Vec().setVectorSize(embLength).setWindowSize(
        5).setNumIterations(10)
    model = word2vec.fit(samples)
    synonyms = model.findSynonyms("158", 20)
    for synonym, cosineSimilarity in synonyms:
        print(synonym, cosineSimilarity)
    embOutputDir = '/'.join(embOutputPath.split('/')[:-1])
    if not os.path.exists(embOutputDir):
        os.makedirs(embOutputDir)
    with open(embOutputPath, 'w') as f:
        for movie_id in model.getVectors():
            vectors = " ".join(
                [str(emb) for emb in model.getVectors()[movie_id]])
            f.write(movie_id + ":" + vectors + "\n")
    embeddingLSH(spark, model.getVectors())
    return model


# def trainItem2vec(spark, samples, embLength, embOutputPath, saveToRedis,
#                   redisKeyPrefix):
#     word2vec = Word2Vec_DF(inputCol="movieIds").setVectorSize(embLength).setWindowSize(
#         5).setMaxIter(10)
#     model = word2vec.fit(samples)
#     vec = model.getVectors()
#     vec.show(5)
#     vec.write.format("json").mode("overwrite").save(embOutputPath[:-4])
#     synonyms = model.findSynonymsArray("158", 20)
#     for synonym, cosineSimilarity in synonyms:
#         print(synonym, cosineSimilarity)
#     embOutputDir = '/'.join(embOutputPath.split('/')[:-1])
#     if not os.path.exists(embOutputDir):
#         os.makedirs(embOutputDir)
#     with open(embOutputPath, 'w') as f:
#         for movie_id in model.getVectors():
#             vectors = " ".join(
#                 [str(emb) for emb in model.getVectors()[movie_id]])
#             f.write(movie_id + ":" + vectors + "\n")
#     embeddingLSH(spark, model.getVectors())
#     return model


def generate_pair(x):
    # eg:
    # watch sequence:['858', '50', '593', '457']
    # return:[['858', '50'],['50', '593'],['593', '457']]
    pairSeq = []
    previousItem = ''
    for item in x:
        if not previousItem:
            previousItem = item
        else:
            pairSeq.append((previousItem, item))
            previousItem = item
    return pairSeq


def generateTransitionMatrix(samples):
    pairSamples = samples.flatMap(lambda x: generate_pair(x))
    pairCountMap = pairSamples.countByValue()
    pairTotalCount = 0
    transitionCountMatrix = defaultdict(dict)
    itemCountMap = defaultdict(int)
    for key, cnt in pairCountMap.items():
        key1, key2 = key
        transitionCountMatrix[key1][key2] = cnt
        itemCountMap[key1] += cnt
        pairTotalCount += cnt
    transitionMatrix = defaultdict(dict)
    itemDistribution = defaultdict(dict)
    for key1, transitionMap in transitionCountMatrix.items():
        for key2, cnt in transitionMap.items():
            transitionMatrix[key1][key2] = \
                transitionCountMatrix[key1][key2] / itemCountMap[key1]
    for itemid, cnt in itemCountMap.items():
        itemDistribution[itemid] = cnt / pairTotalCount
    return transitionMatrix, itemDistribution


def oneRandomWalk(transitionMatrix, itemDistribution, sampleLength):
    sample = []
    # pick the first element
    randomDouble = random.random()
    firstItem = ""
    accumulateProb = 0.0
    for item, prob in itemDistribution.items():
        accumulateProb += prob
        if accumulateProb >= randomDouble:
            firstItem = item
            break
    sample.append(firstItem)
    curElement = firstItem
    i = 1
    while i < sampleLength:
        if (curElement not in itemDistribution) or (
                curElement not in transitionMatrix):
            break
        probDistribution = transitionMatrix[curElement]
        randomDouble = random.random()
        accumulateProb = 0.0
        for item, prob in probDistribution.items():
            accumulateProb += prob
            if accumulateProb >= randomDouble:
                curElement = item
                break
        sample.append(curElement)
        i += 1
    return sample


def randomWalk(transitionMatrix, itemDistribution, sampleCount, sampleLength):
    samples = []
    for i in range(sampleCount):
        samples.append(
            oneRandomWalk(transitionMatrix, itemDistribution, sampleLength))
    return samples


def graphEmb(samples, spark, embLength, embOutputFilename, saveToRedis,
             redisKeyPrefix):
    transitionMatrix, itemDistribution = generateTransitionMatrix(samples)
    print(len(transitionMatrix))
    print(len(itemDistribution))
    sampleCount = 20000
    sampleLength = 10
    newSamples = randomWalk(transitionMatrix, itemDistribution, sampleCount,
                            sampleLength)
    rddSamples = spark.sparkContext.parallelize(newSamples)
    trainItem2vec(spark, rddSamples, embLength, embOutputFilename, saveToRedis,
                  redisKeyPrefix)


def generateUserEmb(spark, rawSampleDataPath, model, embLength, embOutputPath,
                    saveToRedis, redisKeyPrefix):
    ratingSamples = spark.read.format("csv").option("header", "true").load(
        rawSampleDataPath)
    ratingSamples.show(10)
    Vectors_list = []
    for key, value in model.getVectors().items():
        Vectors_list.append((key, list(value)))
    fields = [
        StructField('movieId', StringType(), False),
        StructField('emb', ArrayType(FloatType()), False)
    ]
    schema = StructType(fields)
    Vectors_df = spark.createDataFrame(Vectors_list, schema=schema)
    ratingSamples = ratingSamples.join(Vectors_df, on='movieId', how='inner')
    ratingSamples.show(5)
    result = ratingSamples.select('userId', 'emb').rdd.map(
        lambda x: (x[0], (x[1], 1)))
    result_ = result.collect()
    with open(embOutputPath[:-4] + '_.csv', 'w') as f:
        for row in result_:
            # vectors = " ".join([str(emb) for emb in row[1]])
            vectors = str(row[1])
            f.write(row[0] + ":" + vectors + "\n")
    result = result.reduceByKey(
        lambda a, b: ([a[0][i] + b[0][i] for i in range(len(a[0]))], a[1] + b[1])
    ).map(
        lambda x: (x[0], [x[1][0][i] / x[1][1] for i in range(len(x[1][0]))])
    ).collect()
    with open(embOutputPath, 'w') as f:
        for row in result:
            # vectors = " ".join([str(emb) for emb in row[1]])
            vectors = str(row[1])
            f.write(row[0] + ":" + vectors + "\n")


if __name__ == '__main__':
    print("Start time: "
          + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    conf = SparkConf().setAppName('ctrModel').setMaster('local')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    # Change to your own filepath
    file_path = './src/main/resources'
    rawSampleDataPath = file_path + "/webroot/sampledata/ratings.csv"
    embLength = 10
    samples = processItemSequence(spark, rawSampleDataPath)
    print("Start generating Item2vec...")
    model = trainItem2vec(
        spark, samples, embLength,
        embOutputPath=file_path + "/webroot/modeldata2/item2vecEmb.csv",
        saveToRedis=False,
        redisKeyPrefix="i2vEmb"
    )
    print("Start generating GraphEmb...")
    graphEmb(
        samples, spark, embLength,
        embOutputFilename=file_path + "/webroot/modeldata2/itemGraphEmb.csv",
        saveToRedis=False, redisKeyPrefix="graphEmb"
    )
    print("Start generating UserEmb...")
    generateUserEmb(
        spark, rawSampleDataPath, model, embLength,
        embOutputPath=file_path + "/webroot/modeldata2/userEmb.csv",
        saveToRedis=False,
        redisKeyPrefix="uEmb"
    )
    print("End time: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
