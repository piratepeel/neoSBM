"""runNeoSBM.py - module to run neoSBM on a network with a metadata partition
    Copyright (C) 2016 Leto Peel

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA"""







if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run neoSBM on a network with a metadata partition")
    
    parser.add_argument("networkfile", help="Input network")
    parser.add_argument("metadatafile", help="Metadata labels")
    parser.add_argument("thetamin", help="Minimum value for log10 theta")
    parser.add_argument("-o","--output", help='filename stem for output files')
    parser.add_argument("-s","--sbmopt", help='SBM optimal partition (if known already)')
    parser.add_argument("-p","--plot", help='plots output only (no inference)', action='store_true')
    #~ parser.add_argument("-p","--path", help="Path to files (if not in current directory)")
    args = parser.parse_args()
    
    #~ if args.path:
        #~ path=args.path
    #~ else:
        #~ path="."
    network = args.networkfile.split('.')[0]
    meta = args.metadatafile.split('.')[0]
    
    thetamin=float(args.thetamin)
    
    try:
        assert 10**thetamin < 1
    except AssertionError:
        print '\n\n'
        raise AssertionError("Log10(Thetamin): {} Thetamin: {} \n Thetamin must be less than 1.".format(thetamin,10**thetamin))
        
    
    
    if args.plot:
        import disp_output
        disp_output.plotLq(network,meta,log=True,DC=False)
        disp_output.plt.show()
    
    else :
        import loadNetwork
        
        E,M = loadNetwork.load(args.networkfile, args.metadatafile)

        import neoSBM as ns
        
        if args.sbmopt:
            c=loadNetwork.loadPartition(args.sbmopt)
        else:
            K=len(ns.np.unique(M))
            n=len(M)
            print M
            print K, n
            c = ns.fitSBM(E,K,n,greedy_runs=20)
            ns.writePartition("%s_SBM%i.txt" % (network,K),c)
        
        #ns.run(E,M,c,network,thetamin,sbmModel=SBMmh,iterations=100,logtheta=True,runs=10)
        ns.run(E,M,c,network,meta,thetamin,iterations=100,runs=10)