import pybullet as pyb 
import re
import os
import os.path as p
from pathlib import Path
import sys
from contextlib import contextmanager

physics_client = pyb.connect(pyb.DIRECT)
# pyb.setAdditionalSearchPath(pybullet_data.getDataPath())

def convex_decomposition(file_in, file_out, log="/home/amberxie/log.txt"):
	Path(p.dirname(file_out)).mkdir(parents=True, exist_ok=True)
	pyb.vhacd(str(file_in), str(file_out),  log, convexhullDownsampling=2, concavity=0.002)

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def split_obj_file(file_in, dir_out):
	"""Split an OBJ file into separate files per named object
	Ignores vertex texture coordinates, polygon groups, parameter space vertices.
	The individual files are named as the object they contain. The material file
	(.mtl) is not split with the objects.
	Run: 
	    $ objsplit.py /input/dir/file.obj /output/dir
	    
	Written by Balázs Dukai, https://github.com/balazsdukai
	"""

	v_pat = re.compile(r"^v\s[\s\S]*")  # vertex
	vn_pat = re.compile(r"^vn\s[\s\S]*")  # vertex normal
	f_pat = re.compile(r"^f\s[\s\S]*")  # face
	o_pat = re.compile(r"^o\s[\s\S]*")  # named object
	ml_pat = re.compile(r"^mtllib[\s\S]*")  # .mtl file
	mu_pat = re.compile(r"^usemtl[\s\S]*")  # material to use
	s_pat = re.compile(r"^s\s[\s\S]*")  # shading
	vertices = ['None']  # because OBJ has 1-based indexing
	v_normals = ['None']  # because OBJ has 1-based indexing
	objects = {}
	faces = []
	mtllib = None
	usemtl = None
	shade = None
	o_id = None

	with open(file_in, 'r') as f_in:
		for line in f_in:
			v = v_pat.match(line)
			o = o_pat.match(line)
			f = f_pat.match(line)
			vn = vn_pat.match(line)
			ml = ml_pat.match(line)
			mu = mu_pat.match(line)
			s = s_pat.match(line)

			if v:
				vertices.append(v.group())
			elif vn:
				v_normals.append(vn.group())
			elif o:
				if o_id:
				    objects[o_id] = {'faces': faces,
				                     'usemtl': usemtl,
				                     's': shade}
				    o_id = o.group()
				    faces = []
				else:
					o_id = o.group()
			elif f:
				faces.append(f.group())
			elif mu:
				usemtl = mu.group()
			elif s:
				shade = s.group()
			elif ml:
				mtllib = ml.group()
			else:
			    # ignore vertex texture coordinates, polygon groups, parameter
			    # space vertices
			    pass

		if o_id:
		    objects[o_id] = {'faces': faces,
		                     'usemtl': usemtl,
		                     's': shade}
		else:
			sys.exit("Cannot split an OBJ without named objects in it!")

	# vertex indices of a face
	fv_pat = re.compile(r"(?<= )\b[0-9]+\b", re.MULTILINE)
	# vertex normal indices of a face
	fn_pat = re.compile(r"(?<=\/)\b[0-9]+\b(?=\s)", re.MULTILINE)
	for o_id in objects.keys():
		faces = ''.join(objects[o_id]['faces'])
		f_vertices = {int(v) for v in fv_pat.findall(faces)}
		f_vnormals = {int(vn) for vn in fn_pat.findall(faces)}
		# vertex mapping to a sequence starting with 1
		v_map = {str(v): str(e) for e, v in enumerate(f_vertices, start=1)}
		vn_map = {str(vn): str(e) for e, vn in enumerate(f_vnormals, start=1)}
		faces_mapped = re.sub(fv_pat, lambda x: v_map[x.group()], faces)
		faces_mapped = re.sub(
		    fn_pat, lambda x: vn_map[x.group()], faces_mapped)

		objects[o_id]['vertices'] = f_vertices
		objects[o_id]['vnormals'] = f_vnormals
		# old vertex indices are not needed anymore
		objects[o_id]['faces'] = faces_mapped

	oid_pat = re.compile(r"(?<=o\s).+")
	with suppress_stdout():
		for o_id in objects.keys():
			fname = oid_pat.search(o_id).group()
			file_out = p.join(dir_out, fname + ".obj")
			with open(file_out, 'w', newline=None) as f_out:
				if mtllib:
					f_out.write(mtllib)

				f_out.write(o_id)

				for vertex in objects[o_id]['vertices']:
				    print(vertex)
				    f_out.write(vertices[int(vertex)])

				for normal in objects[o_id]['vnormals']:
					f_out.write(v_normals[int(normal)])

				if objects[o_id]['usemtl']:
					f_out.write(objects[o_id]['usemtl'])

				if objects[o_id]['s']:
					f_out.write(objects[o_id]['s'])

				f_out.write(objects[o_id]['faces'])

if __name__ == '__main__':
	folder = Path("/shared/group/shapenetcore_v2")
	save_folder = Path("/shared/amberxie/shapenet")

	# iterate through synset classes
	synset_classes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
	for synset_class in synset_classes:
		objs = [d for d in os.listdir(os.path.join(folder, synset_class)) if os.path.isdir(os.path.join(folder, synset_class, d))]
		# iterate through objects
		for obj in objs:
			obj_original = folder / synset_class / obj / "models" / "model_normalized.obj" 
			obj_new_folder = save_folder / synset_class / obj
			obj_convex = obj_new_folder / "original" / "model_convex.obj"
			# first, decompose mesh in convex parts
			convex_decomposition(obj_original, obj_convex)
			# save each convex mesh separately
			split_obj_file(obj_convex, obj_new_folder)