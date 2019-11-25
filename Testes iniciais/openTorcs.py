import subprocess
import sys
practice = "practice" + str(sys.argv[1])
p = subprocess.Popen('/home/TCC-JV/torcs-1.3.7/BUILD/bin/torcs -r /home/TCC-JV/torcs-1.3.7/src/raceman/'+ practice +'.xml -nofuel -nodamage')
