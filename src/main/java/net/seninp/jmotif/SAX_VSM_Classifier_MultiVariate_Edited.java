package net.seninp.jmotif;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.atomic.AtomicInteger;

import net.seninp.jmotif.sax.SAXException;
import net.seninp.jmotif.text.Params;
import net.seninp.jmotif.text.TextProcessor;
import net.seninp.jmotif.text.WordBag;
import net.seninp.util.StackTrace;
import net.seninp.util.UCRUtils;
import org.slf4j.LoggerFactory;
import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import com.beust.jcommander.JCommander;

/**
 * This implements a classifier.
 * 
 * @author psenin
 * 
 */
public class SAX_VSM_Classifier_MultiVariate_Edited{

  private static final DecimalFormatSymbols otherSymbols = new DecimalFormatSymbols(Locale.US);
  private static DecimalFormat fmt = new DecimalFormat("0.00###", otherSymbols);
  private static final Object CR = "\n";
  private static final String COMMA = ", ";

  private static TextProcessor tp = new TextProcessor();

  private static List<Map<String, List<double[]>>> trainData = new ArrayList<Map<String, List<double[]>>>();
  private static List<Map<String, List<double[]>>> testData = new ArrayList<Map<String, List<double[]>>>();

  // static block - we instantiate the logger
  //
  private static final Logger consoleLogger;
  private static final Level LOGGING_LEVEL = Level.INFO;

  static {
    consoleLogger = (Logger) LoggerFactory.getLogger(SAXVSMClassifier.class);
    consoleLogger.setLevel(LOGGING_LEVEL);
  }

  public static double Sax_VSM_Classifier_MultiVariate_Test(int alphabet_size,int paa_size, int window_size,
		  List<String> trainDataPath,List<String> testDataPath, List<String> suffixPath) throws SAXException{

    try {
    	trainData.clear();
    	testData.clear();
    	
      SAXVSMClassifierParams params = new SAXVSMClassifierParams();
      params.SAX_ALPHABET_SIZE =alphabet_size;
      params.SAX_PAA_SIZE = paa_size;
      params.SAX_WINDOW_SIZE = window_size;
      params.TEST_FILE = "E:\\UCI HAR Dataset\\test\\Inertial Signals\\body_acc_x_test.csv";
      params.TRAIN_FILE = "E:\\UCI HAR Dataset\\train\\Inertial Signals\\body_acc_x_train.csv";
      
      //JCommander jct = new JCommander(params, args);


      StringBuffer sb = new StringBuffer(1024);
      sb.append("SAX-VSM Classifier").append(CR);
      sb.append("parameters:").append(CR);

      sb.append("  train data:                  ").append(SAXVSMClassifierParams.TRAIN_FILE).append(CR);
      sb.append("  test data:                   ").append(SAXVSMClassifierParams.TEST_FILE).append(CR);
      sb.append("  SAX sliding window size:     ").append(SAXVSMClassifierParams.SAX_WINDOW_SIZE).append(CR);
      sb.append("  SAX PAA size:                ").append(SAXVSMClassifierParams.SAX_PAA_SIZE).append(CR);
      sb.append("  SAX alphabet size:           ").append(SAXVSMClassifierParams.SAX_ALPHABET_SIZE).append(CR);
      sb.append("  SAX numerosity reduction:    ").append(SAXVSMClassifierParams.SAX_NR_STRATEGY).append(CR);
      sb.append("  SAX normalization threshold: ").append(SAXVSMClassifierParams.SAX_NORM_THRESHOLD).append(CR);
      for(int i = 0; i<trainDataPath.size();i++)
      {
    	  trainData.add(UCRUtils.readUCRData(trainDataPath.get(i)));
          consoleLogger.info("trainData classes: " + trainData.size() + ", series length: "
              + trainData.get(i).entrySet().iterator().next().getValue().get(0).length);
          for (Entry<String, List<double[]>> e : trainData.get(i).entrySet()) {
            consoleLogger.info(" training class: " + e.getKey() + " series: " + e.getValue().size());
          }

          testData.add(UCRUtils.readUCRData(testDataPath.get(i)));
          consoleLogger.info("testData classes: " + testData.size() + ", series length: "
              + testData.get(i).entrySet().iterator().next().getValue().get(0).length);
          for (Entry<String, List<double[]>> e : testData.get(i).entrySet()) {
            consoleLogger.info(" test class: " + e.getKey() + " series: " + e.getValue().size());
          } 
      }
      

    }
    catch (Exception e) {
      System.err.println("There was an error...." + StackTrace.toString(e));
      System.exit(-10);
    }
    List<Params> params_List = new ArrayList<>();
    for(int i = 0; i<suffixPath.size();i++)
    {
    	params_List.add(new Params(window_size,
    	        paa_size, alphabet_size,
    	        SAXVSMClassifierParams.SAX_NORM_THRESHOLD, SAXVSMClassifierParams.SAX_NR_STRATEGY,suffixPath.get(i)));
    }
    
    return classify(params_List);
   }

  private static double classify(List<Params> params_list) throws SAXException {
	List<WordBag> bags = null;
    for(int i=0; i<params_list.size();i++)
    {
    	// making training bags collection
    	if(i ==0)
    	{
            bags = tp.labeledSeries2WordBags(trainData.get(i), params_list.get(i));
    	}
    	else
    	{
    		List<WordBag> temp_bags = tp.labeledSeries2WordBags(trainData.get(i), params_list.get(i));
    		for(int j=0; j< bags.size(); j++)
    		{
    			for(int k=0; k<temp_bags.size();k++)
    			{
    				if(temp_bags.get(k).getLabel().equals(bags.get(j).getLabel()))
    				{
    					for (Entry<String, AtomicInteger> word : temp_bags.get(k).getInternalWords().entrySet()) {
    						bags.get(j).addWord(word.getKey(), word.getValue().get());
						}
    				}
    			}
    		}
    	}
    }
	
    // getting TFIDF done
    HashMap<String, HashMap<String, Double>> tfidf = tp.computeTFIDF(bags);
    // classifying
    int testSampleSize = 0;
    int positiveTestCounter = 0;
    
    for (String label : tfidf.keySet()) {
      List<List<double[]>> testD = new ArrayList<>();
      for (Map<String, List<double[]>> testdata_var : testData) {
    	  testD.add(testdata_var.get(label));
	}
      int series_count = testD.get(0).size();
      for (int i =0; i<series_count;i++) {
    	  WordBag test = null;
    	  for(int j =0; j<testD.size();j++)
    	  {   
    		  if(j ==0)
    		  {
        		  test = tp.seriesToWordBag("test", testD.get(j).get(i), params_list.get(j));
    		  }
    		  else
    		  {
    			  WordBag test_temp = tp.seriesToWordBag("test", testD.get(j).get(i), params_list.get(j));
    			  for (Entry<String, AtomicInteger> word : test_temp.getInternalWords().entrySet()) {
    				  test.addWord(word.getKey(),word.getValue().get());
				}
    		  }
    	  }
    	  positiveTestCounter = positiveTestCounter
    	            + tp.classify(label,test, tfidf);
    	        testSampleSize++;
      }
      
      //for (double[] series : testD) {
    //	  WordBag test = seriesToWordBag("test", series, params);
      //  positiveTestCounter = positiveTestCounter
     //       + tp.classify(label, series, tfidf, params);
     //   testSampleSize++;
    //  }
    }

    // accuracy and error
    double accuracy = (double) positiveTestCounter / (double) testSampleSize;
    double error = 1.0d - accuracy;

    // report results
    System.out.println("classification results: " + toLogStr(params_list.get(0), accuracy, error));
    return accuracy;
  }

  protected static String toLogStr(Params params, double accuracy, double error) {
    StringBuffer sb = new StringBuffer();
    sb.append("strategy ").append(params.getNrStartegy().toString()).append(COMMA);
    sb.append("window ").append(params.getWindowSize()).append(COMMA);
    sb.append("PAA ").append(params.getPaaSize()).append(COMMA);
    sb.append("alphabet ").append(params.getAlphabetSize()).append(COMMA);
    sb.append(" accuracy ").append(fmt.format(accuracy)).append(COMMA);
    sb.append(" error ").append(fmt.format(error));
    return sb.toString();
  }

}

