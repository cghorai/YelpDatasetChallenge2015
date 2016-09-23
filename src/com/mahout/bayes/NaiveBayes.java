package com.mahout.bayes;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.StringReader;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.util.Version;
import org.apache.mahout.classifier.naivebayes.NaiveBayesModel;
import org.apache.mahout.classifier.naivebayes.StandardNaiveBayesClassifier;
import org.apache.mahout.classifier.naivebayes.training.TrainNaiveBayesJob;
import org.apache.mahout.classifier.sgd.AdaptiveLogisticRegression;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.vectorizer.SparseVectorsFromSequenceFiles;
import org.apache.mahout.vectorizer.TFIDF;

import com.data.setup.PorterStemmer;
import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Multiset;

public class NaiveBayes {

	Configuration configuration = new Configuration();

	private static final String TRAIN_DATA_PATH = "unigram/TestData/TestDataSetServicesFull.csv";
	private static final String TEST_DATA_PATH = "unigram/TrainingData/TrainDataSetServicesFull.csv";
	private static final String SEQ_FILE_PATH = "input/yelp-seq";
	private static final String LABEL_INDX_PATH = "input/labelindex";
	private static final String MODEL_PATH = "input/model";
	private static final String VECTOR_PATH = "input/yelp-vectors";
	private static final String DICTIONARY_PATH = "input/yelp-vectors/dictionary.file-0";
	private static final String FREQUENCY_PATH = "input/yelp-vectors/df-count/part-r-00000";

	public static void main(String[] args) throws Throwable {
		NaiveBayes naiveBayes = new NaiveBayes();
		
		// Create Model
		naiveBayes.createSequenceFile();
		naiveBayes.createSparseVector();
		
		// Train data on Naive Bayes Model 
		naiveBayes.trainNaiveBayesModel();

		// Read test file and Calculate
		naiveBayes.calculateAndPrintRMSE();

	}

	public void calculateAndPrintRMSE() throws IOException {
		//Define stemmer class
		PorterStemmer st = new PorterStemmer();
		NaiveBayes naiveBayes = new NaiveBayes();
		//Read test data
		BufferedReader reader = new BufferedReader(new FileReader(
				TEST_DATA_PATH));
		
		String line;
		int testedClass;
		int sumOfSquare = 0;
		int actualClass;
		int count = 0;
		
		//Ignore first line
		reader.readLine();
		
		// Traverse test file
		while ((line = reader.readLine()) != null) {
			count++;
			String[] tokens = line.split(",");
			testedClass = naiveBayes.predictRating(st.stripAffixes(tokens[1]));
			actualClass = Integer.valueOf(tokens[0]);

			int err = actualClass - testedClass;
			sumOfSquare = sumOfSquare + err * err;
		}
		reader.close();
		double RMSE = Math.sqrt((double) sumOfSquare / count);
		System.out.println("RMSE : " + RMSE);

	}

	public void createSequenceFile() throws Exception {
		
		// Define stemmer class
		PorterStemmer st = new PorterStemmer();
		BufferedReader reader = new BufferedReader(new FileReader(
				TRAIN_DATA_PATH));
		FileSystem fs = FileSystem.getLocal(configuration);
		Path seqFilePath = new Path(SEQ_FILE_PATH);
		fs.delete(seqFilePath, false);
		
		//Write to this path
		SequenceFile.Writer writer = SequenceFile.createWriter(fs,
				configuration, seqFilePath, Text.class, Text.class);
		
		int count = 0;
		
		
		try {
			String line;
			while ((line = reader.readLine()) != null) {
				String[] tokens = line.split(",");
				writer.append(new Text("/" + tokens[0] + "/yelp" + count++),
						new Text(st.stripAffixes(tokens[1])));  //st.stripAffixes is used for stemming
			}
		} finally {
			reader.close();
			writer.close();
		}
	}

	void createSparseVector() throws Exception {
		SparseVectorsFromSequenceFiles svfsf = new SparseVectorsFromSequenceFiles();
		svfsf.run(new String[] { "-i", SEQ_FILE_PATH, "-o", VECTOR_PATH, "-ow",
				"-ng", "2" }); // Use "-ng", "2" for bigram and "-ng", "3" for trigram. Ignore for unigram 
	}

	void trainNaiveBayesModel() throws Exception {
		TrainNaiveBayesJob trainNaiveBayes = new TrainNaiveBayesJob();
		trainNaiveBayes.setConf(configuration);
		trainNaiveBayes.run(new String[] { "-i",
				VECTOR_PATH + "/tfidf-vectors", "-o", MODEL_PATH, "-li",
				LABEL_INDX_PATH, "-el", "-c", "-ow" });
	}


	private int predictRating(String review) throws IOException {

		
		// Get dictionary, document frequency and words
		Map<String, Integer> dictionary = getDictionary(configuration,
				new Path(DICTIONARY_PATH));
		Map<Integer, Long> documentFrequency = getFrequency(
				configuration, new Path(FREQUENCY_PATH));
		Multiset<String> words = ConcurrentHashMultiset.create();

		// Extract the words using Lucene
		Analyzer analyzer = new StandardAnalyzer(Version.LUCENE_46);
		TokenStream tokenStream = analyzer.tokenStream("text",
				new StringReader(review));
		CharTermAttribute termAttribute = tokenStream
				.addAttribute(CharTermAttribute.class);
		tokenStream.reset();
		int wordCount = 0;
		
		// Compare words with dictionary
		while (tokenStream.incrementToken()) {
			if (termAttribute.length() > 0) {
				String word = tokenStream.getAttribute(CharTermAttribute.class)
						.toString();
				Integer wordId = dictionary.get(word);
				// If the word is not in the dictionary, skip it
				if (wordId != null) {
					words.add(word);
					wordCount++;
				}
			}
		}
		tokenStream.end();
		tokenStream.close();

		// Get count
		int documentCount = documentFrequency.get(-1).intValue();

		// Create vector for the reviews
		Vector vector = new RandomAccessSparseVector(10000);
		TFIDF tfidf = new TFIDF();
		
	
		for (Multiset.Entry<String> entry : words.entrySet()) {
			String word = entry.getElement();
			int count = entry.getCount();
			Integer wordId = dictionary.get(word);
			Long freq = documentFrequency.get(wordId);
			double tfIdfValue = tfidf.calculate(count, freq.intValue(),
					wordCount, documentCount);
			vector.setQuick(wordId, tfIdfValue); // wordId => TFIDF weight
		}

		// Create model : matrix (wordId, labelId)
		NaiveBayesModel model = NaiveBayesModel.materialize(
				new Path(MODEL_PATH), configuration);
		StandardNaiveBayesClassifier classifier = new StandardNaiveBayesClassifier(
				model);

		// Get Score for each classifier and emit the classifier with highest score
		Vector resultVector = classifier.classifyFull(vector);
		double bestScore = -Double.MAX_VALUE;
		int bestClassifierId = -1;
		// System.out.println(resultVector);

		for (Element element : resultVector.all()) {
			int categoryId = element.index();
			double score = element.get();

			if (score > bestScore) {
				bestScore = score;
				bestClassifierId = categoryId;
			}

		}
		// System.out.println(bestCategoryId);

		analyzer.close();
		return bestClassifierId;
	}

	// Read dictionary
	public static Map<String, Integer> getDictionary(Configuration conf,
			Path dictionnaryPath) {
		Map<String, Integer> dictionnary = new HashMap<String, Integer>();
		for (Pair<Text, IntWritable> pair : new SequenceFileIterable<Text, IntWritable>(
				dictionnaryPath, true, conf)) {
			dictionnary.put(pair.getFirst().toString(), pair.getSecond().get());
		}
		return dictionnary;
	}

	// Read frequency
	public static Map<Integer, Long> getFrequency(Configuration conf,
			Path documentFrequencyPath) {
		Map<Integer, Long> documentFrequency = new HashMap<Integer, Long>();
		for (Pair<IntWritable, LongWritable> pair : new SequenceFileIterable<IntWritable, LongWritable>(
				documentFrequencyPath, true, conf)) {
			documentFrequency
					.put(pair.getFirst().get(), pair.getSecond().get());
		}
		return documentFrequency;
	}
}