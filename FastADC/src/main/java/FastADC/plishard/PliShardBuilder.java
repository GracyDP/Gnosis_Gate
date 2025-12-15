package FastADC.plishard;

import com.koloboke.collect.set.hash.HashIntSet;
import com.koloboke.collect.set.hash.HashIntSets;
import de.metanome.algorithms.dcfinder.input.ParsedColumn;
import net.mintern.primitive.Primitive;

import java.util.*;


public class PliShardBuilder {

    private final int shardLength;
    private final boolean[] isNum;

    public PliShardBuilder(int _shardLength, List<ParsedColumn<?>> pColumns) {
        shardLength = _shardLength;
        int colCount = pColumns.size();

        isNum = new boolean[colCount];
        for (int col = 0; col < colCount; col++)
            isNum[col] = pColumns.get(col).getType() != String.class;
    }

    // build the Pli of a given column within tuple id range [beg, end)
    private Pli buildPli(boolean isNum, int[] colValues, int beg, int end) {
        HashIntSet keySet = HashIntSets.newMutableSet();
        for (int row = beg; row < end; ++row) {
            keySet.add(colValues[row]);
//            System.out.println("[PliShardBuilder] buildPli: "+colValues[row]);
        }
//        System.out.println("\n---\n");
        int[] keys = keySet.toIntArray();
        if (isNum)
            Primitive.sort(keys, (a, b) -> Integer.compare(b, a), false);

        Map<Integer, Integer> keyToClusterID = new HashMap<>(); // int (key) -> cluster id
        for (int clusterID = 0; clusterID < keys.length; clusterID++)
            keyToClusterID.put(keys[clusterID], clusterID);

        List<Cluster> clusters = new ArrayList<>();             // put tuple indexes in clusters
        for (int i = 0; i < keys.length; i++)
            clusters.add(new Cluster());
        for (int row = beg; row < end; ++row) {
//            System.out.println("[PliShardBuilder] row: "+row+"-- "+colValues[row]+"  keyToClusterID.get(colValues[row]): "+keyToClusterID.get(colValues[row]));
            clusters.get(keyToClusterID.get(colValues[row])).add(row);
        }

//        System.out.println("new Pli(clusters, keys, keyToClusterID) : "+new Pli(clusters, keys, keyToClusterID));
        return new Pli(clusters, keys, keyToClusterID);
    }

    public PliShard[] buildPliShards(int[][] intInput) {
        if (intInput == null || intInput.length == 0 || intInput[0].length == 0)
            return new PliShard[0];

        int rowBeg = 0, rowEnd = intInput[0].length;
        int nShards = (rowEnd - rowBeg - 1) / shardLength + 1;
        PliShard[] pliShards = new PliShard[nShards];

        for(int i=0; i< nShards; i++) {
            int shardBeg = rowBeg + i * shardLength, shardEnd = Math.min(rowEnd, shardBeg + shardLength);
            List<Pli> plis = new ArrayList<>();
            for (int col = 0; col < intInput.length; col++)
                plis.add(buildPli(isNum[col], intInput[col], shardBeg, shardEnd));
            pliShards[i] = new PliShard(plis, shardBeg, shardEnd);
        }

//        IntStream.range(0, nShards).forEach(i -> {
//            int shardBeg = rowBeg + i * shardLength, shardEnd = Math.min(rowEnd, shardBeg + shardLength);
//            List<Pli> plis = new ArrayList<>();
//            for (int col = 0; col < intInput.length; col++)
//                plis.add(buildPli(isNum[col], intInput[col], shardBeg, shardEnd));
//            pliShards[i] = new PliShard(plis, shardBeg, shardEnd);
//        });

        return pliShards;
    }

}
