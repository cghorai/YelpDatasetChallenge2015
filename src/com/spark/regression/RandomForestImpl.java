package com.spark.regression;

import scala.Tuple2;

import java.util.HashMap;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.mllib.util.MLUtils;

public class RandomForestImpl {
	// Load train and test data files
	private static final String TEST_DATA_PATH = "unigram/TestData/Services_Test_Tfidf.txt";
	private static final String TRAINING_DATA_PATH = "unigram/TrainingData/Services_Train_Tfidf.txt";

	@SuppressWarnings({ "serial", "resource" })
	public static void main(String[] args) {
		// Create Configuration and Spark Context
		SparkConf sparkConf = new SparkConf().setAppName(
				"JavaRandomForestClassification").setMaster("local[2]");
		JavaSparkContext sc = new JavaSparkContext(sparkConf);

		// Load the the in JavaRDD format with LabeledPoint for training and
		// testing
		JavaRDD<LabeledPoint> trainingData = MLUtils.loadLibSVMFile(sc.sc(),
				TRAINING_DATA_PATH).toJavaRDD();
		JavaRDD<LabeledPoint> testData = MLUtils.loadLibSVMFile(sc.sc(),
				TEST_DATA_PATH).toJavaRDD();

		



		// Empty categoricalFeaturesInfo indicates all features are continuous.
		HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();

		// Configuration for training RandomForest
		Integer numberOfClasses = 6;
		Integer numberOfTrees = 30; // Number of trees
		String subsetStrategy = "auto"; // Let the algorithm choose.
		String impurity = "gini"; // gini impurity is used
		Integer maximumDepth = 30;
		Integer maximumBins = 10000;
		Integer seed = 99999;

		// Train RandomForestModel
		final RandomForestModel model = RandomForest.trainClassifier(
				trainingData, numberOfClasses, categoricalFeaturesInfo,
				numberOfTrees, subsetStrategy, impurity, maximumDepth,
				maximumBins, seed);

		// Evaluate model on test data
		JavaPairRDD<Double, Double> predictionAndLabel = testData
				.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
					@Override
					public Tuple2<Double, Double> call(LabeledPoint p) {
						return new Tuple2<Double, Double>(model.predict(p
								.features()), p.label());
					}
				});

		// Calculate Mean Squared Error
		Double meanSquaredError = predictionAndLabel.map(
				new Function<Tuple2<Double, Double>, Double>() {
					@Override
					public Double call(Tuple2<Double, Double> pl) {
						Double diff = pl._1() - pl._2();
						return diff * diff;
					}
				}).reduce(new Function2<Double, Double, Double>() {
			@Override
			public Double call(Double a, Double b) {
				return a + b;
			}
		})/ (testData.count());

		// Print Root Mean Squared Error
		System.out.println("Test Root Mean Squared Error: "
				+ Math.sqrt(meanSquaredError));

	}
}