package net.seninp.jmotif.direct;

/**
 * An entry comparator.
 * 
 * The direct code was taken from JCOOL (Java COntinuous Optimization Library), and altered for
 * SAX-VSM needs.
 * 
 * @see <a href="https://github.com/dhonza/JCOOL/wiki">https://github.com/dhonza/JCOOL/wiki</a>
 *
 */
import java.util.Comparator;
import java.util.Map.Entry;

public class TfIdfEntryComparator implements Comparator<Entry<String, Double>> {

  @Override
  public int compare(Entry<String, Double> arg0, Entry<String, Double> arg1) {
    return -arg0.getValue().compareTo(arg1.getValue());
  }

}
