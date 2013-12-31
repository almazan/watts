function [histogram,lenhoc] = compute_phoc(string,levels,digits,caseSensitive)

if nargin==2
    digits=0;
    caseSensitive = 0;
elseif nargin==3
    caseSensitive = 0;
end

if caseSensitive == 0
    string = lower(string);
end

if caseSensitive
    numchars = 26*2;
else
    numchars = 26;
end
if digits
    numchars = numchars+10;
end

lenhoc=numchars*sum(levels);
histogram = uint8(zeros(lenhoc,1));
i=1;
for p=levels
    e=i+p*numchars-1;
    h = compute_hist(string,p,numchars,digits,caseSensitive);
    histogram(i:e) = h;
    i=e+1;
end

lenhoc = length(histogram);

function hist = compute_hist(string,p,numchars,digits,caseSensitive)
len = length(string);
hist = uint8(zeros(numchars*p,1));
for i=1:p
    b = [(1/p)*(i-1), (1/p)*(i)];
    for c=1:len
        a = [(1/len)*(c-1), (1/len)*(c)];
        pr = prop(a,b);
        if string(c)<=122 && string(c)>=97 % a lowercase character
            idx = string(c)-96 + (i-1)*numchars;
        elseif digits && string(c)<=57 && string(c)>=48 % a number
            idx = string(c)-47 + 26 + (i-1)*numchars;
            if caseSensitive
                idx = idx + 26;
            end
        elseif caseSensitive && string(c)<=90 && string(c)>=65 % an uppercase character
            idx = string(c)-89 + 26 + (i-1)*numchars;
        else % a special character or a character not considered
            idx = -1;
        end
        
        if pr>0.49 && idx>0
            hist(idx) = hist(idx)+1;
        end
    end
end


function p = prop(a,b)
intersection = @(a,b) ~(a(2) < b(1) || a(1) > b(2)) ...
    * ( min(a(2),b(2)) - max(a(1),b(1)));
p= intersection(a,b) / (a(2)-a(1));
