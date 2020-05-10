function vertface2ply(v, f, t, name)
% VERTFACE2OBJ Save a set of vertice coordinates and faces as a Wavefront/Alias Obj file
% VERTFACE2OBJ(v,f,fname)
%     v is a Nx3 matrix of vertex coordinates.
%     f is a Mx3 matrix of vertex indices. 
%     fname is the filename to save the obj file.

fid = fopen(name,'w');

fprintf(fid,'ply\nformat ascii 1.0\ncomment VCGLIB generated\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\nelement face %d\nproperty list uchar int vertex_indices\nend_header\n', size(v, 1), size(f, 1));

for i=1:size(v,1)
	fprintf(fid,'%f %f %f %d %d %d 255\n',v(i,1),v(i,2),v(i,3), t(i,1), t(i,2), t(i,3));
end

for i=1:size(f,1);
	fprintf(fid,'3 %d %d %d\n',f(i,1) - 1,f(i,2) - 1,f(i,3) - 1);
end

fclose(fid);

