function bbox = DoBB(im,px,py,show)
% Gets the smallest bounding box where the vertical projections contains at
% least py of all the pixels and the horizontal projection contains px of
% all the pixels.
% bbox is [y1,y2,x1,x2]

if nargin<4
    show=0;
end

rng(0);

if nargin < 2
    px = 0.975; % Goal is to center. Be conservative
end
if nargin < 3
    py = 0.8; % Goal is to remove ascenders/descenders. Free for all.
end

imbw = 1-im2bw(im,graythresh(im));

sh = (sum(imbw)/sum(sum(imbw)));
sv = (sum(imbw,2)/sum(sum(imbw)));
shc = [0 cumsum(sh)];
svc = [0 cumsum(sv)'];
dh = bsxfun(@minus,shc,shc') > px;
dv = bsxfun(@minus,svc,svc') > py;

[p1h,p2h] = find(dh);
a= randperm(length(p1h)); p1h = p1h(a);p2h = p2h(a);
[p1v,p2v] = find(dv);
a= randperm(length(p1v)); p1v = p1v(a);p2v = p2v(a);

[vh,idxh] = sort(abs(p2h-p1h+1));
[vv,idxv] = sort(abs(p2v-p1v+1));

ph=[p1h(idxh(1)),p2h(idxh(1))];
pv=[p1v(idxv(1)),p2v(idxv(1))];

bbox = [min(ph),max(ph),min(pv),max(pv)];

if show
    imshow(im);
    rectangle('Position',[bbox(1),bbox(3),bbox(2)-bbox(1)+1,bbox(4)-bbox(3)+1],'LineStyle','--','EdgeColor','r')
end
end



