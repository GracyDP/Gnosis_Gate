import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Iterator;

import FastADC.FastADC;
import de.metanome.algorithms.dcfinder.denialconstraints.DenialConstraint;
import de.metanome.algorithms.dcfinder.denialconstraints.DenialConstraintSet;

public class Main {

    public static void main(String[] args) {
        String fp = "./dataset/vehicle0_1_lite.csv";
        double threshold = 0.0d;
        int rowLimit = -1;              // limit the number of tuples in dataset, -1 means no limit
        int shardLength = 1;
        boolean linear = false;         // linear single-thread in EvidenceSetBuilder
        boolean singleColumn = false;   // only single-attribute predicates

        FastADC fastADC = new FastADC(singleColumn, threshold, shardLength, linear);
        DenialConstraintSet dcs = fastADC.buildApproxDCs(fp, rowLimit);
        
        String path_dcs = "./output/DCs_"+fp.split("/")[2].replaceAll(".csv", "")+".txt";
        
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(path_dcs))) {
            Iterator<DenialConstraint> it = dcs.iterator();

            while (it.hasNext()) {
                DenialConstraint dc = it.next();
//                System.out.println("dcs " + dc);
                writer.write(dc.toString());
                writer.newLine();
            }

            System.out.println("Denial constraints salvati nel file "+path_dcs);

        } catch (IOException e) {
            System.err.println("Errore durante il salvataggio dei denial constraints: " + e.getMessage());
        }
    }

}
