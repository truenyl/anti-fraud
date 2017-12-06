# -*- coding: utf-8 -*-
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler
from pyspark.sql.types import StructField, StructType, FloatType, StringType, DoubleType
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import ChiSqSelector
from pyspark.mllib.stat import Statistics
from pyspark.mllib.linalg import Vectors
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.sql.functions import rand
from pyspark.mllib.classification import SVMWithSGD
from pyspark.sql import SparkSession,SQLContext #2.1
from pyspark import SparkConf, SparkContext
import math
import numpy as np

class modeling:
    # dataset - 训练集， estimator - 模型， estimatorParamMaps - 模型所有参数组合， samplingrates - 负样本抽样率list， numfolds - 交叉验证折数
    def _fit(self, dataset, estimator, estimatorParamMaps, samplingrates, numfolds = 5):
        all_list = dataset.columns
        all_list.remove('"Class"') #所有特征列名
        assembler = VectorAssembler().setInputCols(all_list).setOutputCol('features_vector') #特征列转换为一列向量
        labelIndexer = StringIndexer(inputCol='"Class"', outputCol='label') #统一标签列名称为label
        featureIndexer = VectorIndexer(inputCol='features_vector', outputCol='features', maxCategories=10) #统一特征向量列名称，不同值数量小于10视作离散变量编号
        pipeline = Pipeline(stages=[labelIndexer, featureIndexer, estimator]) #机器学习流建模，三部分整合
        dataset = assembler.transform(dataset) #训练集生成特征向量列
        dataset.show()
        best_epm, best_sampling, metricsX = self.Cross_Validation(dataset, pipeline, estimatorParamMaps, samplingrates, numfolds) #交叉验证
        bestModel = pipeline.fit(dataset.sampleBy('"Class"', fractions = {1.0: 1.0, 0.0: best_sampling}),best_epm) # fit最优模型并输出
        return bestModel, best_epm, best_sampling
    def Cross_Validation(self, dataset, estimator, estimatorParamMaps, samplingrates, numFolds):
        est = estimator
        epm = estimatorParamMaps
        numModels = len(epm) #参数组合长度
        sam = samplingrates
        nFolds = numFolds
        h = 1.0 / nFolds
        metricsX = np.zeros(numModels * len(sam))
        for k in range(len(sam)):
            training = dataset.sampleBy('"Class"', fractions = {1.0: 1.0, 0.0: sam[k]}, seed = 0).repartition(1)
            training.cache()
            df = dataset.select('*', rand(0).alias('_rand')) #加入一列随机数
            for i in range(nFolds):
                validateLB = i * h
                validateUB = (i + 1) * h
                condition = (df['_rand'] >= validateLB) & (df['_rand'] < validateUB) #按随机数分折
                validation = df.filter(condition)
                train = df.filter(~condition)
                for j in range(numModels):
                    model = est.fit(train, epm[j])
                    predictions = model.transform(validation, epm[j])
                    tp = predictions.rdd.map(lambda x: x.prediction == 1 and x.label == 1).filter(lambda f: f == True).count() #true positive
                    fp = predictions.rdd.map(lambda x: x.prediction == 1 and x.label == 0).filter(lambda f: f == True).count() #false positive
                    fn = predictions.rdd.map(lambda x: x.prediction == 0 and x.label == 1).filter(lambda f: f == True).count() #false negetive
                    if (tp + fp == 0 or tp == 0):
                        metrics = 0
                    else:
                        p = float(tp) / float(tp + fp)
                        precision = p / (p + (1 - p) * 5.0)       #此处5.0为将抽样率1:4下的数据集精准率转换为1:20数据集上的情况，即（1/4）/（1/20），根据训练集和实际测试情况调整
                        recall = float(tp) / float(tp + fn)       #召回
                        metrics = 2 / ((1 / recall) + (1 / precision))    #F1-score
                    metricsX[k * numModels + j] += metrics
        best = np.argmax(metricsX)
        bestIndex = best % numModels
        best_sampling = sam[best / numModels]
        return epm[bestIndex], best_sampling, metricsX