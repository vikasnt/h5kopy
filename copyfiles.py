import glob
import h5kopy
import h5py

for filename in glob.glob('*.h5'):
	with h5py.File(filename) as f:
		file = h5py.File('copy'+filename,'w')
		h5kopy.copy(f,0,file,0)