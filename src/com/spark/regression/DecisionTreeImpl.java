package com.spark.regression;

import java.util.HashMap;
import java.util.Map;

import scala.Tuple2;

import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.SparkConf;

public class DecisionTreeImpl {
	// Load train and test data files
		private static final String TEST_DATA_PATH = "unigram/TestData/Hotels_Test_Tfidf.txt";
		private static final String TRAINING_DATA_PATH  = "unigram/TrainingData/Hotels_Train_Tfidf.txt";

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
			HashMap<Integer, Integer> categoryFeatures = new HashMap<Integer, Integer>();


		// Decision tree algorithm configuration
		String impurity = "variance";
		Integer maximumDepth = 30;
		Integer maximumBins = 10000;

		// Train the DecisionTree model
		final DecisionTreeModel model = DecisionTree.trainRegressor(
				trainingData, categoryFeatures, impurity, maximumDepth,
				maximumBins);

		// Evaluate model on test data
		JavaPairRDD<Double, Double> predictionAndLabel = testData
				.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
					@Override
					public Tuple2<Double, Double> call(LabeledPoint p) {
						return new Tuple2<Double, Double>(model.predict(p
								.features()), p.label());
					}
				});
		
		//Calculate the Mean Squared Error
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
		System.out.println("Test Root Mean Squared Error: " + Math.sqrt(meanSquaredError));

	}
}