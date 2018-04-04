package net.seninp.jmotif;

import java.util.ArrayList;
import java.util.List;

import net.seninp.jmotif.sax.SAXException;

public class TestRunnerMultiVar {

	
public static void main(String[] args)
{
	List<String> trainDataList = new ArrayList<>();
	List<String> testDataList = new ArrayList<>();
	List<String> suffixList = new ArrayList<>();
	trainDataList.add("E:\\UCI HAR Dataset\\train\\Inertial Signals\\body_acc_x_train.csv");
	trainDataList.add("E:\\UCI HAR Dataset\\train\\Inertial Signals\\body_acc_y_train.csv");
	trainDataList.add("E:\\UCI HAR Dataset\\train\\Inertial Signals\\body_acc_z_train.csv");
	trainDataList.add("E:\\UCI HAR Dataset\\train\\Inertial Signals\\body_gyro_x_train.csv");
	trainDataList.add("E:\\UCI HAR Dataset\\train\\Inertial Signals\\body_gyro_y_train.csv");
	trainDataList.add("E:\\UCI HAR Dataset\\train\\Inertial Signals\\body_gyro_z_train.csv");
	trainDataList.add("E:\\UCI HAR Dataset\\train\\Inertial Signals\\total_acc_x_train.csv");
	trainDataList.add("E:\\UCI HAR Dataset\\train\\Inertial Signals\\total_acc_y_train.csv");
	trainDataList.add("E:\\UCI HAR Dataset\\train\\Inertial Signals\\total_acc_z_train.csv");
	
	testDataList.add("E:\\UCI HAR Dataset\\test\\Inertial Signals\\body_acc_x_test.csv");
	testDataList.add("E:\\UCI HAR Dataset\\test\\Inertial Signals\\body_acc_y_test.csv");
	testDataList.add("E:\\UCI HAR Dataset\\test\\Inertial Signals\\body_acc_z_test.csv");
	testDataList.add("E:\\UCI HAR Dataset\\test\\Inertial Signals\\body_gyro_x_test.csv");
	testDataList.add("E:\\UCI HAR Dataset\\test\\Inertial Signals\\body_gyro_y_test.csv");
	testDataList.add("E:\\UCI HAR Dataset\\test\\Inertial Signals\\body_gyro_z_test.csv");
	testDataList.add("E:\\UCI HAR Dataset\\test\\Inertial Signals\\total_acc_x_test.csv");
	testDataList.add("E:\\UCI HAR Dataset\\test\\Inertial Signals\\total_acc_y_test.csv");
	testDataList.add("E:\\UCI HAR Dataset\\test\\Inertial Signals\\total_acc_z_test.csv");
	
	suffixList.add("_x_acc");
	suffixList.add("_y_acc");
	suffixList.add("_z_acc");
	suffixList.add("_x_gyro");
	suffixList.add("_y_gyro");
	suffixList.add("_z_gyro");
	suffixList.add("_x_total");
	suffixList.add("_y_total");
	suffixList.add("_z_total");
	
	int[] alphabet_arr = new int[]{2,4,5,6};
	int[] paa_arr = new int[]{8,10,12,14,16,18,20,22,25,30};
	int[] window_arr = new int[]{80,60,50,40,35};
	double max_accuracy = 0;
	int paa_best = 0;
	int alphabet_best = 0;
	int window_best = 0;
	for(int i=0; i<alphabet_arr.length;i++)
	{
		for(int j=0;j<paa_arr.length; j++)
		{
			for(int k=0; k<window_arr.length;k++)
			{
				try {
					double temp_accu = SAX_VSM_Classifier_MultiVariate_Edited.Sax_VSM_Classifier_MultiVariate_Test(alphabet_arr[i], paa_arr[j], window_arr[k], trainDataList, testDataList, suffixList);
					if(temp_accu>max_accuracy)
					{
						max_accuracy = temp_accu;
						window_best =window_arr[k];
						paa_best = paa_arr[j];
						alphabet_best = alphabet_arr[i];
					}
				} catch (SAXException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
					System.out.println("best accuracy: "+max_accuracy);
					System.out.println("best window size: "+window_best);;
					System.out.println("best paa size: "+paa_best);;
					System.out.println("best alphabet size: "+alphabet_best);;
				}
				
			}
		}
	}
	System.out.println("best accuracy: "+max_accuracy);
	System.out.println("best window size: "+window_best);;
	System.out.println("best paa size: "+paa_best);;
	System.out.println("best alphabet size: "+alphabet_best);;
}
}
