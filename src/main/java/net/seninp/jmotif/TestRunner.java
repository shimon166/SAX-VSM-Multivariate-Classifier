package net.seninp.jmotif;

import net.seninp.jmotif.sax.SAXException;

public class TestRunner {

	
public static void main(String[] args)
{
	int[] alphabet_arr = new int[]{2,4,5,6,7,8,9,10,11,12};
	int[] paa_arr = new int[]{5,6,7,8,10,12,14,16,18,20};
	int[] window_arr = new int[]{128,80,60,50,40,30,25};
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
					double temp_accu = SAX_VSM_Classifier_Edited.Sax_VSM_Classifier_Test(alphabet_arr[i], paa_arr[j], window_arr[k]);
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
