import subprocess
import sys
practice = "practice" + str(sys.argv[1])
p = subprocess.Popen('C:\\Program Files (x86)\\torcs\\wtorcs.exe -r config\\raceman\\'+ practice +'.xml -nofuel -nodamage')
p.communicate()
