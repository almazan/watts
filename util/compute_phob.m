function [hob,lenhob] = compute_phob(string,bgrams,levels)

string = lower(string);
hob = [];
for p=levels
    hob = [hob compute_hist(string,p,bgrams)];
end
lenhob = length(hob);

function hist = compute_hist(string,p,bgrams)
numB = length(bgrams);
hist = uint8(zeros(numB*p,1));
len = length(string);
if len<2
    return;
end
for i=1:p
    b = [(1/p)*(i-1), (1/p)*(i)];
    for c=1:len-1;
        sub = string(c:c+1);
        a = [(1/len)*(c-1), (1/len)*(c+1)];
        pr = prop(a,b);
        if pr>0.49
            idx = find(strncmp(sub,bgrams,2),1) + (i-1)*numB;
            hist(idx) = hist(idx)+1;
        end
    end
end


function p = prop(a,b)
intersection = @(a,b) ~(a(2) < b(1) || a(1) > b(2)) ...
    * ( min(a(2),b(2)) - max(a(1),b(1)));
p= intersection(a,b) / (a(2)-a(1));
