"""Export portable environment locks, without machine-specific headers."""
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent
result=subprocess.run([str(HERE/'tools/bin/micromamba'),'--no-rc','list','-p',str(HERE/'.env'),'--explicit'],check=True,capture_output=True,text=True)
urls=[line.strip() for line in result.stdout.splitlines() if line.startswith(('https://','http://'))]
if not urls:raise RuntimeError('No package URLs found; refusing to overwrite lock file')
(HERE/'conda-explicit.txt').write_text('# platform: linux-64\n@EXPLICIT\n'+'\n'.join(urls)+'\n')
packages=subprocess.run([sys.executable,'-m','pip','list','--format=freeze'],check=True,capture_output=True,text=True)
(HERE/'requirements-lock.txt').write_text(packages.stdout)
print('Exported portable conda-explicit.txt and requirements-lock.txt')
