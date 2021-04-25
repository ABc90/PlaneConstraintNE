%Qucik Program to demo the use of findPointNormals

%generate a set of 3d points
x = repmat(1:49,49,1);
y = x';
z = peaks;
points = [x(:),y(:),z(:)];

%find the normals and curvature
[normals,curvature] = findPointNormals(points,[],[0,0,10],true);

%plot normals and colour the surface by the curvature
hold off;
surf(x,y,z,reshape(curvature,49,49));
hold on;
quiver3(points(:,1),points(:,2),points(:,3),...
    normals(:,1),normals(:,2),normals(:,3),'r');
axis equal;