package com.data.setup;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.mahout.common.Pair;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.vectorizer.DictionaryVectorizer;
import org.apache.mahout.vectorizer.DocumentProcessor;
import org.apache.mahout.vectorizer.common.PartialVectorMerger;
import org.apache.mahout.vectorizer.tfidf.TFIDFConverter;

public class CreateTFIDFTestFile {


	static final String OUTPUTDOLDER = "output/";
	static final String INPUT_FILE_PATH = "Bigram/TestData/TestDataSetServicesFull.csv";
	

	public static void main(String args[]) throws Exception {
		Configuration configuration = new Configuration();

		FileSystem fs = FileSystem.get(configuration);
		Path sequencePath = new Path(OUTPUTDOLDER, "sequence");
		Path tokenizedPath = new Path(OUTPUTDOLDER,
				DocumentProcessor.TOKENIZED_DOCUMENT_OUTPUT_FOLDER);

		tokenizeDocuments(fs, configuration, sequencePath, tokenizedPath);
		createTFIDF(configuration, tokenizedPath);
		readSequenceFile(fs, configuration, "tfidf/tfidf-vectors",
				"part-r-00000", new VectorWritable());
	}

	public static void readSequenceFile(FileSystem fs,
			Configuration configuration, String path, String file,
			Writable writable) throws Exception {

		Path vectorsFolder = new Path(OUTPUTDOLDER, path);
		SequenceFile.Reader reader = new SequenceFile.Reader(fs, new Path(
				vectorsFolder, file), configuration);

		Text key = new Text();
		Writable value = writable;
		int countLines = 0;

		PrintWriter writer = new PrintWriter("input.txt", "UTF-8");
		// PrintWriter testWriter = new PrintWriter("test_input.txt", "UTF-8");

		while (reader.next(key, value)) {
			countLines++;
			if (writable instanceof VectorWritable) {
				StringBuffer sLine = new StringBuffer(key.toString()
						.replaceAll(":.*", "")
						+ " "
						+ ((VectorWritable) value).get().asFormatString()
								.replaceAll("\\{|\\}", "").replaceAll(",", " "));

				writer.println(sLine);

			}

		}

		reader.close();
		writer.close();

	}

	public static void tokenizeDocuments(FileSystem fs,
			Configuration configuration, Path sequencePath, Path tokenizedPath)
			throws Exception {
		PorterStemmer stemmer = new PorterStemmer();
		BufferedReader reader = new BufferedReader(
				new FileReader(INPUT_FILE_PATH));

		SequenceFile.Writer writer = new SequenceFile.Writer(fs, configuration,
				sequencePath, Text.class, Text.class);
		 int countNumberofLines = 0;
		try {
			String line;
			while ((line = reader.readLine()) != null) {

				String[] tokens = stemmer.stripAffixes(line).split(",");

				writer.append(new Text("" + tokens[0] + ":"
						+ countNumberofLines++), new Text(tokens[1]));
			}
		} finally {
			reader.close();
			writer.close();
		}

		DocumentProcessor.tokenizeDocuments(sequencePath,
				StandardAnalyzer.class, tokenizedPath, configuration);
	}

	public static void createTFIDF(Configuration configuration,
			Path tokenizedPath) throws Exception {

		boolean sequential = false;
		boolean named = false;

		Path wordCount = new Path(OUTPUTDOLDER);
		Path tfidf = new Path(OUTPUTDOLDER + "tfidf");

		Path tfVectors = new Path(OUTPUTDOLDER
				+ DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER);

		// Tokenized document in sequence file
		DictionaryVectorizer.createTermFrequencyVectors(tokenizedPath,
				wordCount, DictionaryVectorizer.DOCUMENT_VECTOR_OUTPUT_FOLDER,
				configuration, 2, // the minimum frequency of the feature
				2, // tuple size -  1 = unigram, 2 = unigram and bigram, 3 =
					
				0.0f, // minValue of log likelihood ratio to used to prune
						// ngrams
				PartialVectorMerger.NO_NORMALIZING, // L_p norm to be
													// computed.
				false, // Use log normalization?
				1, // number of reducers
				100, // chunk Size In Megabytes
				sequential, named);

		Pair<Long[], List<Path>> docFrequenciesFeatures = TFIDFConverter
				.calculateDF(tfVectors, tfidf, configuration, 100);

		TFIDFConverter.processTfIdf(tfVectors, tfidf, configuration,
				docFrequenciesFeatures, 1, // The minimum document frequency.
											
				99, // The max document frequency.
				2.0f, true, // whether to use log normalization
				sequential, named, 1); //number of reducers
	}
}