// version information
debug(str("OpenSCAD ", version()[0], ".", version()[1], ".", version()[2]), color="blue", $debug=true);

// ====================================================================
// constants
// ====================================================================

// 3D printing settings
$fa = 0+3;
$fs = 0+0.3;

// library constants
$inf = 1/0; // infinity
$font = "Hiragino Maru Gothic ProN"; // default font
$debug = false; // enable debug messages

// for command line executions only
$BATCH = false; // executing from command line
$NOW = undef; // execution timestamp

// ====================================================================
// math functions
// ====================================================================

// truncate number (or array of numbers) to d decimal places
function trunc(n, d=2) = let(s=pow(10,d)) is_list(n) ? [for(i=n) floor(i*s)/s] : floor(n*s)/s;

// round number n to d decimal places
function round2(n, d=0) = round(n*10^d)/10^d; 

// round n to the nearest multiple of d
function mof(n, d=2) = round(n/d)*d;

// boolean function for exclusive or
function xor(a, b) = (a&&!b)||(!a&&b);

// hyperbolic functions
function sinh(t) = (exp(t)-exp(-t))/2;
function cosh(t) = (exp(t)+exp(-t))/2;
function tanh(t) = let(e=exp(2*t)) (e-1)/(e+1);

// logistic sigmoid function
function logistic(t) = 1/(1+exp(-t));

// permutation without repetition
function perm(n, r) = r==1 ? n : perm(n-1, r-1) * n;

// combination without repetition
function comb(n, r) = r==1 ? n : comb(n-1, r-1) * n / r;

// geometric inversion
function inv(p, r) = p*(r*r)/(p*p);

// length of perimeter of an ellipse (approximation only): rx, ry are major/minor axes (radius)
function perimeter(rx, ry) = let(ry=ifundef(ry, rx), x=abs(rx), y=abs(ry)) x*y==0 ? 0 : let(d=x-y, s=x+y, h=d*d/s/s) abs(d)<1e-3 ? 2*PI*x : PI*s*(1+h/4+h*h/64+h*h*h/256+25*h*h*h*h/16384);

// golden ratio (phi)
function golden() = 2/(sqrt(5)-1);

// Fibonacci number
function fibonacci(n) = n<2 ? n : fibonacci(n-2) + fibonacci(n-1);

// Fresnel integration used for clothoid generation
fresnel = function(t, n=100, i=0, f, s=[0,0]) i==n ? s : let(g=i==0 ? [1,t/3] : -[f[0]*(4*i-3)/(2*i-1)/(4*i+1),f[1]*(4*i-1)/(2*i+1)/(4*i+3)]*t*t/i/2) fresnel(t, n, i+1, g, s+g);

// ====================================================================
// utility functions
// ====================================================================

// modulo function suitable for indexing an array of size k even when n is negative
function mod(n, k) = let(m=n%k) m<0 && k>=0 ? m+k : m;

// constraint integer n to the range [min,k-1]
function curb(n, k, min=0) = n<min ? min : k && n>=k ? k-1 : n;

// constraint real number a to the range [min,max]
function confine(a, min=0, max=1) = a<min ? min : a>max ? max : a;

// derive fa variable that divides 45º evenly for an ellipse, rx,ry=axes, s=max segment length
function _fa(rx, ry, s=$fs) = max(45/floor(perimeter(rx, ifundef(ry, rx))/8/s), 45/ceil(45/$fa));

// derive fn variable based on radius - should be same as system-derived $fn except minimum is 8 instead of 5
function _fn(r, n=8) = $fn ? $fn : ceil(max(min(360/$fa, abs(r)*2*PI/$fs), n));

// lower resolution versions of _fn()
function _fn2(r, n=8) = $fn ? $fn : ceil(max(min(360/$fa, abs(r)*2*PI/$fs)/2, n));
function _fn3(r, n=8) = $fn ? $fn : ceil(max(min(360/$fa, abs(r)*2*PI/$fs)/3, n));

// check if a is defined
function has(a) = (a!=undef);

// provide default value for undefined variable, beware that exp gets evaluated regardless so it should be a constant
function ifundef(a, exp) = (a!=undef ? a : exp);

// provide default value for NaN variable
function ifnan(a, exp) = (a==a ? a : exp);

// check if value falls inclusively inside a range, undef means unlimited
function within(a, l, h) = (l==undef ? true : a>=l) && (h==undef ? true : a<=h);

// check if a is within the inclusive range [b-e,b+e]
function vicinity(a, b, e=0.001) = a>=b-e && a<=b+e;

// interpolate between a and b, return a if s=0, b if s=1
function interp(a, b, s) = a*(1-s) + b*s;

// location almost reaching a from b with a gap of distance d
function almost(a, b, d) = let(s=d/norm(a-b)) a*(1-s) + b*s;

// for handling array type arguments: (1) reuse same scalar (2) wrap around if no df (3) df if no a[idx]
function opt(a, idx, df) = is_list(a) ? df==undef ? wrap(a, idx) : ifundef(a[idx], df) : (a==undef ? df : a);

// generate and echo a random integer seed only if undefined
function rnd_seed(seed) = seed!=undef ? seed : tee(floor(rands(1000, 9999, 1)[0]), "seed", $debug);

// random function that can handle undef seed (workaround for rands() returning undef if seed is undef)
function rnd(min=0, max=1, dim=0, seed) = let(s=rnd_seed(seed), d=ceil(dim), r=rands(min, max, max(1,d), s)) d>0 ? r : r[0];

// random integer of range [min,max] inclusive, default is [0,99]
function rndi(min, max, seed) = floor(rnd(ifundef(min, 0), ifundef(max, 99)+1, 1, seed)[0]);

// random float of range (min,max) exclusive(?), default is (0,1)
function rndf(min, max, seed) = rnd(ifundef(min, 0), ifundef(max, 1), 1, seed)[0];

// random 3D unit vector
function rndv() = unit(rands(-1, 1, 3));

// normalize a vector
function unit(v) = let(n=norm(v)) n==0 ? v : v/n;

// generate a number sequence, begin and end are inclusive
function seq(begin, end, delta) = let(d=(delta!=undef?delta:sign(end-begin))) [for (i=[begin:d:end]) i];

// return a for-loop range [begin:delta:end] using a span definition = [low,high,number of steps]
function range(span) = [span[0]:(span[1]-span[0])/span[2]:span[1]+0.0001];

// return a for-loop range in [0,max) equally divided into number of parts
// set end=1 to produce a closed range [0,max] (0.9999 means not closed), use max to scale the results
function quanta(parts=100, start=0, end=0.9999, max=1) = parts<1 ? [] : let(s=round2(start,3), e=round2(end,3), d=(e-s)/parts) [((s-start)%1==0?s:s+d)*max:d*max:((e-end)%1==0?e+d/2:end)*max];

// return a lookup table of n points for parametric function fn between [0,1] inclusive
function scaler(fn, n) = n<2 ? [] : [let(k=n-1) for (i=[0:k]) let(t=i/k) [t,fn(t)]];

// cycle through a list of colors, randomly if seed is given
function palette(i, seed) = let(c=["red","green","blue","brown","white","purple","yellow","pink","cyan","black","orange"], k=len(c), m=mod(i,k)) seed ? c[floor(rnd(0,k,m+1,seed)[m])] : c[m];

// text styled with HTML font color if not in batch mode
function strc(s, color="red") = $NOW ? s : str("<font color='", color, "'>", s, "</font>"); 

// format array contents as a single string (one element per line)
function strn(a, i=0) = is_list(a) ? i>=len(a) ? "\n" : str("\n", i, ": ", a[i], strn(a, i+1)) : str(a);

// execute a schema: a parametric function returning its domain if the argument is undefined
// example: resolve(function(t) t==undef ? [0:$fa:360-$fa] : [cos(t),sin(t)]*10)
function resolve(schema) = is_function(schema) ? [for (t=schema()) schema(t)] : schema;

// evaluate a unit trace function {fn:t->[x,y]} in n equal steps, close=include final point
function locus(fn, n=10, close=true, reverse=false, arg=undef) = [for (i=reverse?[n:-1:(close?0:1)]:[0:(close?n:n-1)]) arg==undef ? fn(i/n) : fn(i/n, arg)];

// graph a unit parametric function {fn:t->y} in n equal steps as {[t,y]}, close=include final point
function graph(fn, n=10, scale=[1,1], close=true, reverse=false, arg=undef) = [let (s0=opt(scale, 0, 1), s1=opt(scale, 1, 1)) for (i=reverse?[n:-1:(close?0:1)]:[0:(close?n:n-1)]) let(t=i/n) [t*s0,(arg==undef?fn(t):fn(t, arg))*s1]];

// generate a new parametric function which is product of two parametric functions
fxf = function(fn1, fn2) function(t) fn1(t)*fn2(t);

// helper for setting a default viewpoint when file is opened, e.g.
// $vpt = vp([0,0,0.5]);
// $vpr = vp([55,0,-25]);
// $vpd = vp(300); // $vpd should be changed last
function vp(q) = $vpd==140 ? q : undef;

// ====================================================================
// array manipulations
// ====================================================================

// return the rank of array, e.g. rank(undef) = 0, rank(1) = 0, rank([1]) = 1, rank([[1]]) = 2, rank([1,2,3]) = 1
function rank(array, i=0) = is_list(array) ? rank(array[0], i+1) : i;

// return a range of ascending/descending indices
function incline(array) = [0:len(array)-1];
function decline(array) = [len(array)-1:-1:0];
function indices(array, invert=false) = invert ? [len(array)-1:-1:0] : [0:len(array)-1];

// return individual indices of an array, less=skip some at the end
function keys(array, less=0) = let(k=len(array)-less) k>0 ? [for (i=[0:k-1]) i] : [];

// return the key of the first min/max element
function key_min(array) = search(min(array), array)[0];
function key_max(array) = search(max(array), array)[0];

// check if e is in array
function contains(array, e) = len(search(e, array)) > 0;

// ensure e is a list (array)
function enlist(e) = is_list(e) ? e : [e];

// if an array, change length to n by trimming or padding; otherwise, convert to an array of length n
function redim(array, n, pad=0) = n>0 ? let(a=enlist(array)) [for (i=[0:n-1]) ifundef(a[i], pad)] : [];

// get element by wrapping around index
function wrap(array, idx) = array[mod(idx, len(array))];

// get element by applying mod() or curb() on index which can be out of range
function elem(array, idx, loop=false) = loop ? array[mod(idx, len(array))] : array[curb(idx, len(array))];

// return a subarray of nearby elements at idx within +/-r range
function nearby(array, idx, r=1, loop=true) = let(k=len(array)) [for (i=[idx-r:idx+r]) if (loop) array[mod(i, k)] else if (i>=0 && i<k) array[i]];

// split array into n parts (uniformity not optimal)
function split(array, n) = [let (k=len(array), m=ceil(k/n)) for (j=[0:n-1]) [for (i=[m*j:m*j+m-1]) if (i<k) array[i]]];

// rotate array elements in a cyclic fashion (forward for positive n)
function cyclic(array, n) = n==0 ? array : [let(k=len(array)) for (i=[n:n+k-1]) array[mod(i, k)]];

// generate an array containing n copies of e
function repeat(e, n=2) = [for (i=[1:n]) e];

// reduce rank of an array by promoting second-level elements to top, i.e. [[a,b],[c]] to [a,b,c]
function flatten(array) = [for (i=[0:len(array)-1]) each array[i]];

// remove a list of indices from array
function omit(array, list=[]) = [for (i=[0:len(array)-1]) if (len(search(i, list))==0) array[i]];

// remove a specific value from the array
function eliminate(array, value) = [for (e=array) if (e!=value) e];

// return one of the dimensions of an array, e.g. only z values of a set of 3D points, s=index shifting
function slice(array, idx, s=0) = [for (i=[0:len(array)-1]) wrap(array[i], idx+i*s)];

// affix an extra dimension to array elements, e.g. affix([[1,0],[2,5]], [7,8]) -> [[1,0,7],[2,5,8]], tail is cyclic
function affix(array, tail) = tail==undef ? array : [for (i=[0:len(array)-1]) append(array[i], opt(tail, i))];

// resize array by picking one out of every n elements (size increases due to duplicates if n < 1), s=shift
function every(array, n=1, s=0, center=false) = n==1 && s==0 ? array : let(k=len(array), c=floor((k-1)/2), s=center?c%n:s%k) [if (k>n && n>0) for (i=[s:n:k-1]) let(j=round(i)) if (j<k) array[j]];

// swap two elements in an array
function swap(array, i, j) = [for (k=[0:len(array)-1]) array[k==i?j:k==j?i:k]];

// replace one element
function override(array, idx, value) = [for (i=incline(array)) i == idx ? value : array[i]];

// offset one element by delta amount
function plus(array, idx, delta) = [for (i=incline(array)) i == idx ? array[i]+delta : array[i]];

// negate one element
function negate(array, idx) = override(array, idx, -array[idx]);

// remove t elements from tail, and h elements from head
function snip(array, t=1, h=0) = let(k=len(array)) t+h>0 ? [if (k>t+h) for (i=[h:k-t-1]) array[i]] : array;

// add one element at the head
function prepend(array, e) = let(k=len(array)) k==undef ? [e,array] : [for (i=[0:k]) i==0 ? e : array[i-1]];

// add one element at the tail
function append(array, e) = is_list(array) ? concat(array, [e]) : [array,e];

// return subarray
function subarray(array, start=0, end=-1) = let(k=len(array)) [if (k>start) for (i=[min(k-1,start>=0?start:k+start):min(k-1,end>=0?end:k+end)]) array[i]];

// reverse elements, loop=keep first element position
function reverse(array, enable=true, loop=false) = enable ? let(k=len(array), n=loop?k:k-1) [for (i=[0:k-1]) array[(n-i)%k]] : array;

// return the (last-n)th element
function last(array, n=0) = array[len(array)-n-1];

// compute total of an array, up to the i-th element
function sum(array, i) = i==undef ? sum(array, len(array)-1) : i>0 ? array[i] + sum(array, i-1) : array[0];

// compute average of an array, up to the i-th element
function avg(array, i) = i==undef ? avg(array, len(array)-1) : i>0 ? (array[i] + i*avg(array, i-1))/(i+1) : array[0];

// compute accumulated sums
function accum(array, i=0, s=0) = i==len(array) ? [s] : concat([s], accum(array, i+1, s+array[i]));

// return min and max
function minmax(array) = [min(array),max(array)];

// return difference between min and max
function span(array) = max(array) - min(array);

// offset all elements by same amount
function shift(array, by) = [for (i=incline(array)) array[i]+(is_list(by)?by[i]:by)];

// scale all elements by ratio, if ratio is a list then scale respectively (expects same length as array)
function zoom(array, ratio) = [for (i=incline(array)) array[i]*(is_list(ratio)?ratio[i]:ratio)];

// apply one factor per dimension, e.g. shear(array, [1,-1]) will reflect 2D points along y-axis
function shear(array, factors) = [for (e=array) [for (i=incline(e)) e[i]*factors[i]]];

// unique pairs of all elements (when range>0 include only pairs with indices no greater than range apart)
function pairs(array, range=0) = let(k=len(array)-1) [for (i=[0:k], j=[0:k]) if (i>j && (range==0||range>(i-j))) [array[i],array[j]]];

// ====================================================================
// 3D path functions
// ====================================================================

// convert to a 2D point
function as2d(point=0, y=0) = [ifundef(point[0], point), ifundef(point[1], y)];

// check if a path looks like a loop with evenly spaced points
function loopish(path) = let(n=len(path)-1) n>3 && norm(path[n]-path[0])<min(norm(path[n]-path[n-1]),norm(path[1]-path[0]))*2;

// append first point to the end if not already there
function close_loop(path, enable=true) = enable && path[0]!=path[len(path)-1] ? concat(path, [path[0]]) : path;

// remove last point if it's the same as the first
function unloop(path, enable=true) = enable ? let(k=len(path)-1) path[0]==path[k] ? [for (i=[0:k-1]) path[i]] : path : path;

// align path so that segment i is colinear with vector v (default to z-axis)
function plumb(path, i=0, v=[0,0,1]) = let(k=len(path), i=(i+k)%k) (k<2||i>k-2) ? path : path * m3_rotate(v, path[i+1]-path[i]);

// list indices of adjacent duplicate points, e=threshold
function seams(path, loop=false, e=0.01) = let(k=len(path)) [for (i=[0:k-(loop?1:2)]) if (norm(path[i]-path[(i+1)%k])<=e) i];

// eliminate adjacent duplicate points, e=threshold [private: i, q, f]
function fuse(path, loop=false, e=0.01, i, q, f) = let(k=len(path)) k==0 ? path : let(i=ifundef(i, k-1), q=ifundef(q, loop?path[0]:undef), p=path[i], d=!q||norm(p-q)>e) i==0 ? [if (d||!f) p] : concat(fuse(path, loop, e, i-1, p, d?1:f), d?[p]:[]);

// extend a path by adding length h to head and t to tail at consistent directions
function elong(path, h=1, t=1) = let(k=len(path)) k<2 || (h==0 && t==0) ? path : let(v1=path[0]-path[1], v2=path[k-1]-path[k-2], n1=norm(v1), n2=norm(v2)) concat([path[1]+v1*(n1+h)/n1], [if (k>2) for (i=[1:k-2]) path[i]], [path[k-2]+v2*(n2+t)/n2]);

// a list of accumulated lengths along path for each point + last entry is the total length as a loop
function mileage(path, i=0, s=0) = let(k=len(path)) i>k ? [] : concat([s], mileage(path, i+1, s+norm(path[(i+1)%k]-path[i%k])));

// find minimum and maximum segment lengths in path
function seg_length(path, i=0, b, t) = i>len(path)-2 ? [b,t] : let(s=norm(path[i+1]-path[i])) seg_length(path, i+1, b==undef ? s : min(b, s), t==undef ? s : max(t, s)); 

// calculate total length of a path in 3D space
function path_length(path, i=0) = i>=len(path)-1 ? 0 : norm(path[i+1]-path[i]) + path_length(path, i+1);

// like lookup() but for both 2D and 3D paths (non-proportional/snaps to existing points) 
// t in [0,1], e = snap threshold, disable if zero
function path_lookup(path, t, loop=false, e=0) = let(p=close_loop(path, loop), k=len(p), n=k-1, i=floor(t*n), j=t*n-i) i<0||i>=n||j<e ? p[curb(i,k)] : p[i] + j*(p[i+1]-p[i]);

// find array index (as a real number) along path where distance is d
// e.g. return 1.25 if path is [[0,0],[10,0],[30,0]] and d=15; 2 if d=30; or undef if d<0 or d>30
function path_where(path, d=0, i=0) = i<len(path)-1 ? let(s=norm(path[i]-path[i+1])) d>0 && d>s ? path_where(path, d-s, i+1) : (d<0 ? undef : i+d/s) : (abs(d)<0.001 ? i : undef);

// similar to mileage() but also include point indices for lookup
function path_map(path, loop=false, i=0, s=0) = let(p=i==0?close_loop(path, loop):path) concat([[s,i]], i==len(p)-1 ? [] : path_map(p, loop, i+1, s+norm(p[i+1]-p[i])));

// combine a list of 3D paths end to end
function path_concat(paths=[], i=0) = i>=len(paths) ? [] : let(p=paths[i]) concat(i>0 ? subarray(p, 1) : p, shift3d(path_concat(paths, i+1), last(p)-(paths[i+1] ? paths[i+1][0] : [0,0,0])));

// return a subpath based on offsets h and t in mm (+ve relative to start; -ve relative to end; zero means no change)
// e.g. subpath(p, 1, -1) to trim off 1mm at each end; subpath(p, -2, 0) to remove all but the last 2mm
function subpath(path, h=0, t=0) = h==0 && t==0 ? path :
  let(m=path_map(path), k=m[len(m)-1][0], i1=lookup(h<0?k+h:h, m), i2=lookup(t<=0?k+t:t, m))
  let(j1=ceil(i1), j2=floor(i2), f1=i1%1, f2=i2%1) i1>=i2 ? [] : concat([if (h!=0) path[j1]*f1+path[j1-1]*(1-f1)], [if (h==0||t==0||j1<j2) for (j=[j1:j2]) path[j]], [if (t!=0) path[j2]*(1-f2)+path[j2+1]*f2]);

// shorten length of path by at least d, rounded off to the nearest vertex
function shorten(path, d=0, i=0) = d>0 && i<len(path) ? shorten(path, d-norm(path[i]-path[i+1]), i+1) : snip(path, i);

// resample path to increase or decrease number of points to n, without preserving original points
function resample(path, n, loop=true) = let(n=ifundef(n, len(path))) n<=0 ? path : let(q=close_loop(path, enable=loop), k=len(q), mg=mileage(q), d=len(q[0]), mp=[for (j=[0:d-1]) [for (i=[0:k-1]) [mg[i], q[i][j]]]]) [for (t=quanta(n, max=mg[k-1], end=loop?0.9999:1)) [for (s=[0:d-1]) lookup(t, mp[s])]];

// create a straight path between two points with equal-length segments shorter than or equal to ds {see ruler_path()}
function bridge(p1, p2, ds) = let(v=p2-p1, m=ceil(norm(v)/ds)) [for (t=quanta(m)) p1+t*v];

// preserving cardinal, polish a path by taking average of neighbouring points, r=neighbourhood range (+/-r)
function polish(path, r=1, loop=true) = r<1 ? path : [for (i=incline(path)) let(s=nearby(path, i, r, loop), u=s[0]-s[1], v=s[2]-s[1], d=abs(u*v)/(u*u)/(v*v)) avg(nearby(path, i, r+confine(ceil(d*2), 0, 0), loop=loop))];

// refine a path by subdividing long segments into ones shorter than ds, preserving original points
function refine(path, ds, loop) = let(ds=ifundef(ds, $fs*4)) ds<0.02 ? path : let(loop=(loop!=undef?loop:loopish(path)), p=close_loop(path, loop), k=len(p)-1) snip([for (i=[0:k]) each i==k ? [p[k]] : let(l=norm(p[i+1]-p[i])) l<=ds ? [p[i]] : bridge(p[i], p[i+1], ds)], loop?1:0);

// simplify a path by shifting points until each segment length is at least ds
function coarse(path, ds=1, loop=true) = let(p=close_loop(path, enable=loop)) [for (t=quanta(floor(path_length(p)/ds), end=loop?0.9999:1)) path_lookup(p, t)];

// simplify a path by combining colinear segments, f=colinearity (max is 10)
function lean(path, f=1, loop, i=0, k) = let(k=ifundef(k, len(path)), loop=(loop!=undef?loop:loopish(path))) i==k ? loop ? [] : [path[k-1]] : let(p=path[i], u=p-path[(i+k-1)%k], v=path[(i+1)%k]-p) concat(colinear(u, v, 1-0.0006*f) && (loop || i>0) ? [] : [p], lean(path, f, loop, i+1, k));

// preserving original shape as much as possible, generate one with vertices distrubuted more evenly, f=resolution
function uniform(path, f=$fs, loop=true) = f==0 ? path : coarse(refine(path, ds=f/3, loop=loop), ds=f, loop=loop);

// soften a path by replacing each sharp corner with an arc of radius no greater than r
function soften(path, r=5, loop=true, i=0, m, pp) = r==0 ? path :
  let(p=i==0 && loop ? concat([path[len(path)-1]], path, [path[0]]) : path)
  let(m=ifundef(m, len(p)), pp=ifundef(pp, loop ? [] : [p[0]])) i==m-2 ? loop ? pp : concat(pp, [p[m-1]]) :
  let(o=p[i+1], u=p[i]-o, v=p[i+2]-o, mu=norm(u), mv=norm(v), du=u/mu, dv=v/mv, c2=du*dv, t=sqrt((1-c2)/(1+c2)))
  let(a=min([mu/2,mv/2,r/t]), f=o+a*du, g=o+a*dv, c=o+norm([a,min([r,a*t])])*unit(du+dv), e=cross(u,v))
  soften(p, r, loop, i+1, m, concat(pp, abs(e)<1 || a<=0 || mu+mv<$fs ? [o] : e<0 ? ccw_path(f, g, po=c) : cw_path(f, g, po=c)));

// wobble a path's x and y coordinates wrt origin
function wobble(path, by=0.03, n=32) = [for (p=path) let(t=1+cos(n*atan2(p[1], p[0]))*by) has(p[2]) ? [p[0]*t, p[1]*t, p[2]] : p*t];

// wobble a path's y (if 2D) or z coordinate (if 3D)
function roller(path, by=3, n=32) = [let(k=len(path)) for (i=incline(path)) let(p=path[i]) has(p[2]) ? [p[0], p[1], p[2]+cos(n*360*i/k)*by] : [p[0], p[1]+cos(n*360*i/k)*by]];

// compute vector at point i of path radiating away from origin, r = sampling range
function radiate_at(path, i, r=1, loop=true) = let(w=nearby(path, i, r, loop)) avg([for (j=[0:r*2-1]) w[j+1]+w[j]]);

// compute tangent vector at point i of path, r = sampling range
function tangent_at(path, i, r=1, loop=true) = let(w=nearby(path, i, r, loop)) avg([for (j=[0:r*2-1]) w[j+1]-w[j]]);

// compute a normal vector at point i of path, r = sampling range
function normal_at(path, i, r=1, loop=true, c) = let(v=path[i]-(c!=undef?c:centroid3d(path))) proj2(v, unit(tangent_at(path, i, r, loop)));

// compute angle at vertex i of path
function angle_at(path, i, loop=true) = let(w=force2d(nearby(path, i, 1, loop))) angle2d(w[2]-w[1], w[0]-w[1]);

// ====================================================================
// smooth functions using cubic spline
// ====================================================================

// smooth a path by replacing each segment with a bezier curve, div = subdivisions (auto if omit)
function smooth(path, div, loop) = path==undef ? undef : div!=undef&&div<2 ? path : let(n=len(path)) n<3 ? path : n>200 ? echo(strc("smooth(): path too complex"), n=n) [] : loop || loop==undef && loopish(path) ? smooth_loop(path, div, n) : smooth_arc(path, div, n);

// compute bezier curve using 4 points [begin, control1, control2, end], t in [0,1]
function bezier(points, t) = [let(d=len(points[0])) for (i=[0:d-1]) [pow(1-t,3), 3*t*pow(1-t,2), 3*t*t*(1-t), pow(t,3)]*slice(points, i)];

// --------------------------------------------

// subroutine used by smooth_arc() to extract calculation results
function spline_backtrack_arc(b, c, r, p, i) = let(q=(r[i]-c[i]*p)/b[i]) i==0 ? [q] :
  concat(spline_backtrack_arc(b, c, r, q, i-1), [q]);

// subroutine used by smooth_arc() to generate input matrix
function spline_matrix_arc(k, a=[0], b=[2], c=[1], r, i=1) = let(r=ifundef(r, [k[0]+2*k[1]]))
  i==len(k)-2 ? [append(a, 2), append(b, 7), append(c, 0), append(r, 8*k[i]+k[i+1])] :
  spline_matrix_arc(k, append(a, 1), append(b, 4), append(c, 1), append(r, 4*k[i]+2*k[i+1]), i+1);

// subroutine used by smooth_arc() to find first set of control points
function spline_control_points_arc(k, a, b, c, r, i=0) =
  i==0 ? let(m=spline_matrix_arc(k)) spline_control_points_arc(k, m[0], m[1], m[2], m[3], 1) :
  i==len(r) ? let(p=r[i-1]/b[i-1]) concat(spline_backtrack_arc(b, c, r, p, i-2), [p]) :
  let(m=a[i]/b[i-1], bb=override(b, i, b[i]-m*c[i-1]), rr=override(r, i, r[i]-m*r[i-1]))
  spline_control_points_arc(k, a, bb, c, rr, i+1);

// smooth an arc k by replacing each segment with a bezier curve, div = number of subdivisions
function smooth_arc(k, div, n, i=0, p1, p2) = div&&div<=1 ? k : i==n-1 ? [k[i]] :
  let(p1=(p1!=undef?p1:spline_control_points_arc(k)))
  let(p2=(p2!=undef?p2:[for (i=incline(p1)) i==len(p1)-1 ? (k[i+1]+p1[i])/2 : 2*k[i+1]-p1[i+1]]))
  let(d=(div!=undef?div:ceil(path_length([for (t=quanta(8, end=1)) bezier([k[i], p1[i], p2[i], k[i+1]], t)]))))
  concat([for (t=quanta(d)) bezier([k[i], p1[i], p2[i], k[i+1]], t)], smooth_arc(k, div, n, i+1, p1, p2));

// --------------------------------------------

// subroutine used by smooth_loop() to extract calculation results
function spline_backtrack_loop(b, c, r, g, p, i, pn) = let(q=(r[i]-c[i]*p - g[i]*pn)/b[i]) i==0 ? [q] :
  concat(spline_backtrack_loop(b, c, r, g, q, i-1, pn), [q]);

// subroutine used by smooth_loop() to generate input matrix
function spline_matrix_loop(k, w, n, a=[], b=[], c=[], r=[], i=0) = i==n ? [a,b,c,r] :
let(j=(i+1)%n, v=i==0?w[n-1]:w[i-1], u=v+w[i], f=w[i]/w[j], g=u*u*k[i] + v*v*(1+f)*k[j])
  spline_matrix_loop(k, w, n, append(a, w[i]*w[i]), append(b, 2*v*u), append(c, v*v*f), append(r, g), i+1);

// subroutine used by smooth_loop() to find first set of control points
function spline_control_points_loop(k, w, n, sc, a, b, c, r, g, i) =
  i==undef ? let(m=spline_matrix_loop(k, w, n), g=concat([m[0][0]], repeat(0, n-1))) 
  spline_control_points_loop(k, w, n, m[2][n-1], m[0], m[1], m[2], m[3], g, 0) :
  i==n-2 ? let(m=a[n-1]/b[i], b=plus(b, n-1, -m*c[i]), r=plus(r, n-1, -m*r[i]), p=r[n-1]/b[n-1])
  concat(spline_backtrack_loop(b, c, r, g, p, i, p), [p]) :
  let(m=a[i+1]/b[i], s=sc/b[i]) spline_control_points_loop(k, w, n, -s*c[i], 
      plus(a, n-1, n<4 ? -s*c[i] : 0), 
      plus(plus(b, i+1, -m*c[i]), n-1, -s*g[i]),
      plus(c, i+1, i>n-3 ? -m*g[i] : 0), 
      plus(plus(r, i+1, -m*r[i]), n-1, -s*r[i]), 
      override(g, i+1, -m*g[i]), i+1);

// smooth a loop by replacing each segment with a bezier curve, div = number of subdivisions
function smooth_loop(k, div, n, i=0, w, p1, p2) = div&&div<=1 ? k : i==n ? [] :
  let(w=(w!=undef?w:[for (i=[0:n-1]) max(1, norm(k[i]-k[(i+1)%n]))]))
  let(p1=(p1!=undef?p1:spline_control_points_loop(k, w, n)))
  let(p2=(p2!=undef?p2:[for (i=[0:n-1]) let(j=(i+1)%n) k[j]*(1+w[i]/w[j]) - p1[j]*w[i]/w[j]]))
  let(d=(div!=undef?div:round(path_length([for (t=quanta(8, end=1)) bezier([k[i], p1[i], p2[i], k[(i+1)%n]], t)]))))
  concat([for (t=quanta(d)) bezier([k[i], p1[i], p2[i], k[(i+1)%n]], t)], smooth_loop(k, div, n, i+1, w, p1, p2));

// compute control points for smooth_loop() - for debug only
function smooth_loop_cp(k) =
  let(n=len(k), w=[for (i=[0:n-1]) max(1, norm(k[i]-k[(i+1)%n]))], p1=spline_control_points_loop(k, w, n))
  [p1, [for (i=[0:n-1]) let(j=(i+1)%n) k[j]*(1+w[i]/w[j]) - p1[j]*w[i]/w[j]]];

// ====================================================================
// 2D functions
// ====================================================================

// cartesian-to-polar coordinates conversion [x,y]->[a,d]
function c2p(x, y) = is_list(x) ? [atan2(x[1], x[0]), norm(x)] : [atan2(y, x), norm([x,y])];

// polar-to-cartesian coordinates conversion, [a,d]->[x,y] (accepts 
function p2c(a, d) = is_list(a) ? a[1]*[cos(a[0]),sin(a[0])] : d*[cos(a),sin(a)];

// convert a scalar or a 2D point to 3D, filling in y and z as necessary, preserving points already in 3D
function as3d(point, y=0, z=0) = [ifundef(point[0], point), ifundef(point[1], y), ifundef(point[2], z)];

// check for a valid 2D point
function valid2d(point) = is_num(point[0]) && is_num(point[1]);

// make points 2D by discarding higher dimensions
function force2d(points) = is_list(points[0]) ? [for (p=points) [p[0],p[1]]] : [points[0],points[1]];

// check if two vectors are mutually perpendicular, e=tolerance
function perpend2d(u, v, e=1e-4) = u[0]*v[0] + u[1]*v[1] < e;

// check if two vectors are in parallel, e=tolerance
function parallel2d(u, v, e=1e-4) = abs(u[0]*v[1] - u[1]*v[0]) < e;

// angle between two vectors
function angle2d(u, v) = atan2(v*[-u[1],u[0]], v*u);

// rotate CCW 90 degrees about origin
function orth2d(points) = is_list(points[0]) ? [for (p=points) [-p[1],p[0]]] : [-points[1],points[0]];

// scale each coordinate individually {see shear()}
function scale2d(points, ratios) = let(xs=opt(ratios,0), ys=opt(ratios,1,1)) [for (p=points) [p[0]*xs,p[1]*ys]];

// fit points into a rectangle at origin of size dm (zero length means don't care), prop=preserve ratio
// if dm is a scalar, fit points into a circle of diameter dm at origin
function fit2d(points, dm=[50,50], prop=true, center=true) =
  let(d=size2d(points)) min(d)==0 ? center2d(points, enable=center) :
  rank(dm)==0 ? let(e=encircle2d(points)) shift2d(points*dm/e[1]/2, center?-e[0]:[0,0]) :
  let(dx=opt(dm,0), dy=opt(dm,1), px=dx<0?-1:1, py=dy<0?-1:1)
  let(rx=abs(dx)/d[0], ry=abs(dy)/d[1], rr=(rx==0&&ry==0)?1:rx==0?ry:ry==0?rx:min(rx,ry))
  let(sx=prop||dx==0?rr:rx, sy=prop||dy==0?rr:ry) center2d([for (p=points) [p[0]*px*sx,p[1]*py*sy]], enable=center);

// spin 2D points counterclockwise about origin (with right-angle optimization)
function spin2d(points, a=90) = a==0 ? points : let(aa=a%360)
  aa==90 || aa==-270 ? [for (p=points) [-p[1],p[0]]] :
  aa==180 || aa==-180 ? [for (p=points) [-p[0],-p[1]]] :
  aa==270 || aa==-90 ? [for (p=points) [p[1],-p[0]]] :
  points * m2_rotate(a);

// shift 2D points
function shift2d(points, delta=[0,0]) = [for (p=points) [p[0]+delta[0],p[1]+delta[1]]];

// move 2D points to be just above a baseline parallel to the x-axis
function float2d(points, baseline=0) = let(k=-min(slice(points, 1))) shift2d(points, [0,k+baseline]);

// shift 2D points to be flush/center against x-axis and/or y-axis wrt origin
// xsign, ysign: undef=no change, 0=center, -ve/+ve=which side to go
function flush2d(points, xsign, ysign, origin=[0,0]) =
  let(xx=minmax(slice(points, 0)), yy=minmax(slice(points, 1)))
  let(dx=origin[0] - (xsign == undef ? 0 : xsign == 0 ? avg(xx) : xsign > 0 ? xx[0] : xx[1]))
  let(dy=origin[1] - (ysign == undef ? 0 : ysign == 0 ? avg(yy) : ysign > 0 ? yy[0] : yy[1]))
  shift2d(points, [dx,dy]);

// centroid of 2D points (same as average)
function centroid2d(points) = avg(points);

// among many points, find the index of one closest to c (if c is undefined, find the one closest to centroid)
function near2d(points, c) = let(c=(c!=undef?c:avg(points))) key_min([for (p=points) norm(p-c)]);

// center 2D points
function center2d(points, at=[0,0], enable=true) = is_list(points) && enable ? let(c=[for (i=[0:1]) avg(minmax(slice(points, i)))]-at) [for (p=points) p-c] : points;

// deviate 2D points randomly, each coordinate change has range (-max/2,max/2)
function shake2d(points, max=1, seed) = let(k=len(points)) k==0 ? [] : let(s=rnd_seed(seed)) [for (i=[0:k-1]) points[i] + rnd(-max/2, max/2, 2, i+s/(i+1))];

// a list of random 2D points, each coordinate within range (min,max)
function random2d(n, min=-1, max=1, seed) = let(s=rnd_seed(seed)) [for (i=[1:n]) rnd(min, max, 2, i+s/i)];

// a random walk, each step coordinates change within range (min,max)
function wander2d(n, min=-1, max=1, seed, trail=[[0,0]], base=[0,0]) = n>0 ? let(s=rnd_seed(seed)) wander2d(n-1, min, max, s, concat(trail, [trail[len(trail)-1]+rnd(min, max, 2, n+s/n)+base]), base) : trail;

// a random profile, d=diameter, min/max=number of corners, f=resolution, s=softness
function random_profile(d=50, min=4, max=7, f=1, s=5, seed) = let(r=rnd_seed(seed))
  let(w=shake2d(poly_path(d/2, min+rndi(0, max-min+1, seed=r)), d/(s+1), seed=9999-r))
  fit2d(uniform(soften(w, s, loop=true), f, loop=true), d);

// generate a path by rotating a shorter one n times around the origin
function radiate2d(path, n, i=1) = i<n ? concat([let(r=m2_rotate(360/n*i)) for (p=path) p*r], radiate2d(path, n, i+1)) : path;

// combine a list of 2D paths end-to-end, a single point in the list is a vector, from=override first point
function concat2d(paths=[], from, i=0) = let(s=paths[i]) s[0]==undef ? [] : let(p=is_list(s[0])?s:[[0,0],s], d=ifundef(from, p[0])-p[0], k=len(p)) concat([for (i=[(i==0?0:1):k-1]) p[i]+d], concat2d(paths, p[k-1]+d, i+1));

// similar to concat2d except that each step is a single vector instead of a path
function step2d(vectors, from=[0,0], i=0, m) = let(m=ifundef(m, len(vectors)), p=i==0?from:vectors[i-1]+from) concat([p], i==m ? [] : step2d(vectors, p, i+1, m));

// join paths end-to-end to form a loop (will drop the last point if same as the first)
function loop2d(paths=[], from) = let(c=fuse(concat2d(paths, from), loop=true)) snip(c, c[0]==c[len(c)-1]?1:0); 

// extend a path by concatenating a scaled copy of itself across x-axis and/or y-axis (default is x-axis)
function mirror2d(path, xs=-1, ys=1) = concat(path, [for (p=reverse(path)) [p[0]*xs,p[1]*ys]]);

// remove points outside of y-range (l=low, h=high)
function band2d(points, l, h) = [for (p=points) if (within(p[1], l, h)) p];

// remove points outside of retangular range (see spot2d())
function frame2d(points, x1, x2, y1, y2) = [for (p=points) if (within(p[0], x1, x2) && within(p[1], y1, y2)) p];

// remove points outside a circle of radius r at center (see frame2d())
function spot2d(points, r, center=[0,0]) = [for (p=points) if (norm(p-center) <= r) p];

// find intersection of two line segments s1=[p0,p1] and s2=[p2,p3], including end points
function meet2d(s1, s2, virtual=false) = let(r=s1[1]-s1[0], s=s2[1]-s2[0], c=r && s ? cross(r, s) : 0) c==0 ? undef : let(d=s2[0]-s1[0], t=cross(d, s)/c, u=cross(d, r)/c) virtual || within(t, 0, 1) && within(u, 0, 1) ? s1[0]+t*r : undef;

// find intersection of two line segments s1=[p0,p1] and s2=[p2,p3], excluding end points, e=tolerance
function cut2d(s1, s2, e=0) = let(r=s1[1]-s1[0], s=s2[1]-s2[0], c=r && s ? cross(r, s) : 0) c==0 ? undef : let(d=s2[0]-s1[0], t=cross(d, s)/c, u=cross(d, r)/c) (t!=0 && t!=1 && u!=0 && u!=1 && t>-e && t<1+e && u>-e && u<1+e) ? s1[0]+t*r : undef;

// preserving vertices, align the starting point of profile to where dimension d changes sign (0=x, 1=y)
// note that a profile not changing sign in dimension d will fail
function refit2d(profile, d=1) = let(k=len(profile), j=[for (i=[k-1:-1:0]) let(q1=profile[i][d], q2=profile[(i+1)%k][d]) if (q1==0 || (q1<0 && q2>=0)) i][0]) cyclic(profile, abs(profile[j][d])>abs(profile[(j+1)%k][d])? (j+1)%k : j);

// check if profile is strictly convex
function convex2d(profile, f=0.001, i=0, k) = let(k=ifundef(k, len(profile))) i<k ? let(p0=profile[(i+k-1)%k], p1=profile[i], p2=profile[(i+1)%k]) convex2d(profile, f, i+1, k) && cross(unit(p2-p1), unit(p0-p1))>-f : true;

// flip a 2D profile from the xy-plane to the xz-plane in 3D, for debugging throw() and lathe()
function upright2d(profile, shift=[0,0]) = [for (p=shift2d(profile, shift)) [p[0],0,p[1]]];

// squeeze near the center of a profile, x,y=amount
function squeeze2d(profile, x=0, y=0) = let(
    b=box2d(profile),
    c=[b[2][0]+b[0][0],b[2][1]+b[0][1]]/2,
    s=[b[2][0]-b[0][0],b[2][1]-b[0][1]]/2)
  [for (p=profile) let(dx=p[0]-c[0], dy=p[1]-c[1], px=dx/s[0], py=dy/s[1]) c+[
    dx*(px<-1||px>1 ? 1 : 1-(cos(py*180)+1)*x/2),
    dy*(py<-1||py>1 ? 1 : 1-(cos(px*180)+1)*y/2)
  ]];

// normal vector at vertex biased towards the longer side, d = length of vector
function vnormal2d(path, idx, d=1) = let(g=len(path)) g<3 ? undef : let(
    v1 = path[idx]-path[(idx+g-1)%g], 
    v2 = path[(idx+1)%g]-path[idx], 
    m1 = norm(v1),
    m2 = norm(v2),
    n1 = [-v1[1],v1[0]]/m1, 
    n2 = [-v2[1],v2[0]]/m2, 
    w = norm(n1+n2),
    r = max(m1,m2)/min(m1,m2), 
    n = n1+n2+(v2-v1)/r)
  n && w>0 && norm(n)>0.1 ? unit(n)*d : [0,0];

// find intersection point(s) of 2 circles
function circles_meet(c1, c2, r1, r2) = let(d=norm(c1-c2), a=(r1*r1-r2*r2+d*d)/(2*d), m=c1+a*(c2-c1)/d, s=sqrt(r1*r1-a*a)*orth2d(c2-c1)/d) [m+s,m-s];

// find intersection point(s) on the surfaces of 3 spheres
function trilaterate(c1, c2, c3, r1, r2, r3) =
  let(v21=c2-c1, v31=c3-c1, ex=v21/norm(v21), d1=ex*v31, t=v31-d1*ex, ey=t/norm(t))
  let(ez=cross(ex, ey), d2=ey*v31, n21=norm(c2-c1))
  let(x=(r1*r1-r2*r2+n21*n21)/(2*n21), y=(r1*r1-r3*r3-2*d1*x+d1*d1+d2*d2)/(2*d2))
  let(k=r1*r1-x*x-y*y) k<0 ? undef : let(z=sqrt(k)) [c1+x*ex+y*ey+z*ez,c1+x*ex+y*ey-z*ez];

// return [origin, radius] of the circle passing through 3 points in 2D
function circle3p(p1, p2, p3) = (p1==p2 || p2==p3 || p3==p1) ? undef :
  (p2[1]-p1[1])/(p2[0]-p1[0])==(p3[1]-p2[1])/(p3[0]-p2[0]) ? undef :
  let(d12=p1-p2, d13=p1-p3, d31=p3-p1, d21=p2-p1)
  let(x13=p1[0]*p1[0]-p3[0]*p3[0], y13=p1[1]*p1[1]-p3[1]*p3[1])
  let(x21=p2[0]*p2[0]-p1[0]*p1[0], y21=p2[1]*p2[1]-p1[1]*p1[1])
  let(a=-(x13*d12[1]+y13*d12[1]+x21*d13[1]+y21*d13[1])/(d31[0]*d12[1]-d21[0]*d13[1])/2)
  let(b=-(x13*d12[0]+y13*d12[0]+x21*d13[0]+y21*d13[0])/(d31[1]*d12[0]-d21[1]*d13[0])/2)
  let(o=[a,b]) [o, norm(p1-o)];

// return a shortest counterclockwise arc passing through 3 points in 2D
function arc3p(p1, p2, p3) = let(o=circle3p(p1, p2, p3)[0])
let(p=[p1,p2,p3], d=[norm(p2-p3),norm(p1-p3),norm(p1-p2)], m=key_max(d), e1=p[(m+1)%3], e2=p[(m+2)%3])
  o==undef ? [e1,e2] : area2d([e1,p[m],e2])>0 ? ccw_path(e1, e2, po=o) : ccw_path(e2, e1, po=o);

// repair self-intersecting profile, r=radius i.e. if defined, only compare segment i to segments [i-r,i+r],
// lean=may remove points, e=tolerance of intersection detection
// (1) scan for self-intersections to find loops g:{[begin index, end index, intersection point]}
// (2) eliminate proper sub-loops, resulting in a minimal set d
// (3) replace points inside each loop with the intersection point
// algorithm doesn't work in all cases and slow O(n^2): needs some serious optimization
function tidy2d(p, r, loop=false, lean=false, e=0.001) = r==0 ? p : let(k=len(p))
  let(g=[for (i=[1:k], j=[1:k]) if (i<j && (r==undef || j-i<r || i+k-j<r)) let(c=cut2d([p[i%k],p[(i+1)%k]], [p[j%k],p[(j+1)%k]], e)) if (c && (loop || j+1!=k)) j-i<k/2+1 ? [i+1,j,c] : [j+1,i+k,c]])
  let(d=[for (i=g) if (len([for (j=g) if (j!=i && ((j[0]<=i[0] && j[1]>=i[1])||(j[0]<=i[0]+k && j[1]>=i[1]+k))) 1])==0) i])
  [for (i=[0:k-1]) let(t=[for (j=d) if (within(i,j[0],j[1]) || within(i+k,j[0],j[1])) j]) if (!lean || len(t)==0 || i==t[0][1]%k) len(t)==0 ? p[i] : t[0][2]];

// return a profile keeping distance d away from the given profile, tidy=optional radius for cleanup
// note that number of points on the resulting profile will be different from the original
function escort2d(profile, d=1, loop=false, tidy, i=0, m, e, p) = d==0 ? profile :
  let(m=ifundef(m, len(profile))) i==(loop ? m+1 : m) ? tidy2d(p, tidy, loop=loop, lean=true) :
  let(p1=profile[(i+m-1)%m], p2=profile[i%m], u=unit([p2[1]-p1[1],p1[0]-p2[0]])*d)
  let(v1=p1+u, v2=p2+u, n=(loop&&i==1)||e&&norm(e[1]-v1)>$fs/2, v=e?abs(d)*unit((e[1]+v1)/2-p1)+p1:undef)
  let(pp=i==0 ? [] : concat(p, [if (n&&(loop||i>1)) v], [if (n||i==1) v1, v2]))
  escort2d(profile, d, loop, tidy, i+1, m, [v1,v2], pp);

// inflate/deflate (offset) a profile (does not work well in some cases) [private: i, m, e, p]
// it's different from escort2d() because it always assumes a loop and preserves number of points in the original
function offset2d(profile, d=1, f=3, tidy=50, i=0, m, e, p) = d==0 ? profile :
  let(m=ifundef(m, len(profile))) i==m+1 ?  tidy2d(p, tidy, loop=true, lean=false) :
  let(s1=profile[(i+m-1)%m], s2=profile[i%m], u=unit([s2[1]-s1[1],s1[0]-s2[0]])*d, v1=s1+u, v2=s2+u)
  let(pp=i==0 ? [] : concat(p, [v1==v2 ? e[1] : colinear(e[1]-e[0], v2-v1) ? v1 : let(c=ifundef(meet2d(e, [v1,v2], true), v1)) abs(norm(c-s1)/d)>f ? s1*0.95+c*0.05 : c])) offset2d(profile, d, f, tidy, i+1, m, [v1,v2], pp);

// produce a profile surrounding path, t=thickness, rounded=circular ends (loop=true causes a hack to create a "hole")
function fence2d(path, t=1, rounded=true, s=0, loop=false, tidy) = len(path)<2 ? [] :
  let(p=rectify(path), rounded=(rounded && t>$fs), s=min(s, t/2))
  let(b1=escort2d(p, t/2, tidy=0, loop=loop), b2=escort2d(p, -t/2, tidy=0, loop=loop))
  let(s1=close_loop(soften(tidy2d(b1, tidy, lean=true), s, loop=loop), loop))
  let(s2=close_loop(reverse(soften(tidy2d(b2, tidy, lean=true), s, loop=loop), loop=loop), loop))
  let(e1=!loop && rounded ? subarray(ccw_path(last(s2), s1[0], po=p[0]), 1, -2) : [])
  let(e2=!loop && rounded ? subarray(ccw_path(last(s1), s2[0], po=last(p)), 1, -2) : [])
  concat(s1, e2, s2, e1);

// produce a scaled profile by shifting each point a constant distance away from origin (see also scale2d, offset2d)
function expand2d(profile, by) = [for (p=profile) let(n=norm(force2d(p)),s=(n+by)/n) p[2]==undef ? p*s : [p[0]*s,p[1]*s,p[2]]];

// return the bounding box (4 points) of a profile, b=border
function box2d(profile, b=0) = let(x=minmax(slice(profile,0)), y=minmax(slice(profile,1))) [[x[0]-b,y[0]-b],[x[1]+b,y[0]-b],[x[1]+b,y[1]+b],[x[0]-b,y[1]+b]];

// width of bounding box
function box2dw(profile) = span(slice(profile,0));

// height of bounding box
function box2dh(profile) = span(slice(profile,1));

// center of bounding box
function box2dc(profile) = [avg(slice(profile,0)), avg(slice(profile,1))];

// return [width,height] of a profile
function size2d(profile) = len(profile)<2 ? [0,0] : [span(slice(profile,0)),span(slice(profile,1))];

// compute the [origin, radius] of a non-optimal enclosing circle for a profile
function encircle2d(profile, n=16) = let(v=[for (a=quanta(n, max=90)) let(p=spin2d(profile, a), s=size2d(p)) [abs(s[0]-s[1]),avg(spin2d(box2d(p), -a))]], c=v[key_min(slice(v,0))][1]) [c,max([for (p=profile) norm(p-c)])];

// return the signed area of a profile (+ve means counterclockwise)
function area2d(profile, i=0) = let(k=len(profile)) i>=k ? 0 : let(p1=profile[i], p2=profile[(i+1)%k]) (p1[0]-p2[0])*(p1[1]+p2[1]) + area2d(profile, i+1);

// rectify the orientation of a profile (to be counterclockwise wrt z), center=relocate to origin
function rectify(profile, dm=[0,0], center=false) = let(p=fit2d(profile, dm=dm, prop=true, center=center)) reverse(p, enable=area2d(p)<0);

// a sign derived from 3 vectors indicating if they follow the right-hand-rule (+ve=yes, -ve=no, 0=degenrate case)
function sign3v(v1, v2, v3) = v1==undef||v2==undef||v3==undef ? 0 : cross(v1, v2) * v3;

// Ulam spiral on xy plane, counterclockwise starting with ulam(1) = [0,1,0], ulam(2) = [-1,1,0], etc.
function ulam(n) = let(
    r = floor((sqrt(n)-1)/2)+1,
    c = (2*r-1)*(2*r-1),
    p = (n-c-4*r+1)/(2*r),
    x = n==0 ? 0 : p>=-1 && p<=0 ? -r : p>=1 ? r : p<=0 ? r-(p+2)*2*r : -r+p*2*r,
    y = n==0 ? 0 : p>=0 && p<=1 ? -r : p<=-1 ? r : p<=0 ? r-(p+1)*2*r : -r+(p-1)*2*r)
  [x, y, 0];

// points on a grid of width w and depth d, divided evenly by nw and nd segments respectively
function grid(w, d, nw=2, nd=2) = [for (i=quanta(nw, end=1), j=quanta(nd, end=1)) [i*w-w/2,j*d-d/2]]; 

// isometric grid of width w and depth d, s=spacing between any two adjacent points
function iso_grid(w, d, s=1, center=true) = [let(r=sqrt(3)/2, m2=floor(w/s/2), n1=floor(d/s/r), n2=floor(d/s/r/2)) for (j=center?[-n2:n2]:[0:n1], i=center?j%2==0?[-m2:m2]:[-floor((w+s)/s/2):floor((w-s)/s/2)]:[0:(j%2==0?floor(w/s):floor((w-s/2)/s))]) [i*s+(j%2==0?0:s/2), j*s*r]];

// m by n hex grid where each cell has depth=1
function hex_cell(i, j) = let(r=sqrt(3)/2) i%2==0 ? [i*r,j] : [i*r,j+0.5];
function hex_grid(m, n, trim=true) = let(n=ifundef(n, m)) [for (i=[0:m-1], j=[0:(trim && i%2==0?n-1:max(0,n-2))]) hex_cell(i, j)];

// ====================================================================
// R^2 profiles (counterclockwise wherever applicable)
// ====================================================================

function quad_path(w, d, center=true) = let(d=ifundef(d,w)) center ? let(x=w/2, y=d/2) [[x,y],[-x,y],[-x,-y],[x,-y]] : [[0,0],[w,0],[w,d],[0,d]]; // 4 corners only
function rect_path(x, y) = let(y=ifundef(y,x)) [[x,y],[-x,y],[-x,-y],[x,-y]]; // 4 corners only
function round_path(x, y, a=[0,360]) = let(y=y?y:x, k=confine(a[1]-a[0], -360, 360), n=ceil(_fn(max(abs(x),abs(y)))*abs(k)/360), m=abs(k)<360?n:n-1) [for (i=[0:m]) let(t=a[0]+k*i/n) [x*cos(t),y*sin(t)]];
function fillet_path(x=1, y, convex=false, loop=true) = let(y=y==undef?x:y, e=x*y==0, n=max(1,round(90/_fa(x,y)))) concat([if (loop&&!e) [0,0]], [for (i=[0:n]) let(t=90*i/n) convex ? [x*cos(t),y*sin(t)] : [x-x*sin(t),y-y*cos(t)]]);
function spiral_path(d, a=[0,360], f=3, start=0, end=1) = let(a0=ifundef(a[0],0), a1=opt(a,1), aa=a1-a0) [for (t=quanta(ceil(PI*d*abs(aa)*(end-start)/600/$fs), start=start, end=end)) let(b=a0+t*aa, e=pow(t,f)*d/2) [cos(b)*e,sin(b)*e]];

// ====================================================================
// R^2 designer profiles (diameter d)
// ====================================================================

function arc_path(d=[1,1], a=[0,180]) = let(x=opt(d,0)/2, y=opt(d,1)/2, k=confine(a[1]-a[0], -360, 360), n=ceil(_fn(max(abs(x),abs(y)))*abs(k)/360), m=min(x,y)==0?0:abs(k)<360?n:n-1) [for (i=[0:m]) let(t=a[0]+k*i/n) [cos(t)*x,sin(t)*y]];
function ring_path(d=10, a=[0,360], s=[1,1]) = let(k=confine(a[1]-a[0], -360, 360), n=ceil(_fn(d/2)*abs(k)/360), m=abs(k)<360?n:n-1) [for (i=[0:m]) let(t=a[0]+k*i/n) [cos(t)*s[0],sin(t)*s[1]]*d/2];
function apple_path(d=10) = let(n=ceil(_fn(d/2)*1.5)) [for (i=[0:n-1]) apple_trace(i/n)*d];
function puffy_path(d=10) = radiate2d(shift2d(round_path(d, a=[-24,24], $fa=$fa/2), [-d/2,0]), 4);
function bang_path(d=10) = radiate2d(shift2d(round_path(0.1875*d, a=[224,136]), [0.625*d,0]), 12);
function floral_path(d=10, n=5) = let(c=n+1) [for (t=quanta(_fn2(d)*sqrt(n))) [c*cos(360*t)-cos(c*360*t),c*sin(360*t)-sin(c*360*t)]*d/(2*c)];
function petal_path(d=10, n=5, c=0) = [for (t=quanta(_fn(d*n)*2, max=360)) [cos(t),sin(t)]*max(c,(d/3+d/5*cos(180+n*t)))];
function hump_path(d=10, n=12, f=0.2) = radiate2d([for (t=quanta(ceil(_fn3(PI*d/n/2)), max=360/n)) [cos(t),sin(t)]*d*(1+f*abs(sin(t*n/2)))/2], n);
function heart_path(d=10) = [for (t=quanta(_fn(d/2))) let(s=abs(2*t-1), r=0.7*d*(s*s*s-7*s*s+10*s)/(5-pow(s,12))) [r*cos(360*t)-d/5,r*sin(360*t)]];
function poly_path(d=10, n=5) = [for (t=quanta(n)) let(a=180/n+360*t) [cos(a),sin(a)]*d/2];
function star_path(d=10, n=5, f=4) = [for (i=[0:n*2-1]) let(a=180/n+180*i/n) [cos(a),sin(a)]*d/(i%2?f:2)];
function wave_path(d=10, n=7, f=5, b=0) = [for (t=quanta(_fn(d*n)*2)) let(a=360*t) [cos(a),sin(a)]*(d/2-d*f/50+(cos(180+n*a)+b*sin(n*a*2-90))*d*f/50)];
function butterfly_path(d=10) = let(n=_fn(d/2)*2) [for (i=[0:n-1]) butterfly_trace(i/n)*d];
function egg_path(d=10) = let(n=_fn(d/2)) [for (i=[0:n-1]) egg_trace(i/n)*d];
function box_path(w=10, d) = let(d=ifundef(d,w)) [for (t=quanta(_fn(min(w,d)/16)*8)) box_trace(t, [w,d], true)];
function arch_path(d=10) = [for (t=quanta(_fn(d))) arch_trace(t)-[0.5,0.5]]*d;
function tear_path(d=10, f=1) = [for (a=quanta(_fn(d/3), max=360)) [(cos(a)-1)*f*d+d, (cos(a)+1)*sin(a)*d/3]/2];
function teeth_path(d=10, n=20, f=1, s=1) = let(a=90/n) radiate2d(concat(ring_path(d, [-a+s,a-s]), ring_path(d-f, [a+s,a*3-s])), n); // f=depth of teeth, s=teeth sharpness
function puzzle_path(d=10, y=0) = [for (t=quanta(_fn(d/2), end=1)) [d/2,d+y]-puzzle_trace(t)*d];
function comma_path(d=10, f=0.5) = let(n=_fn2(d/2)) [for (t=quanta(n, end=1, max=1)) [cos(t*360),sin(t*360)]*(1-t+f*t)*d/2];
function leaf_path(d, f) = let(f=ifundef(f, d*0.6)) [for (a=quanta(_fn(d/3), max=360)) [d*cos(a),f*sin(a)^3]/2];
function cloud_path(w, d, n=11, f=3, seed) = let(d=ifundef(d, w), seed=rnd_seed(seed), p=shake2d(round_path(w/2, d/2, $fn=n), (w+d)/40, seed=seed), k=len(p)) loop2d([for (i=[0:k-1]) ccw_path(p[i], p[(i+1)%k], f)]);
function pad_path(w, d, r=2, half=false) =
  let(d=ifundef(d,w), r=max(0.01, min(r, min(w, d)/2-0.01)), w2=w/2, d2=d/2) concat(
    ccw_path([w2,d2-r], [w2-r,d2], po=[w2-r,d2-r]),
    ccw_path([-w2+r,d2], [-w2,d2-r], po=[-w2+r,d2-r]),
    half ? [[-w/2,-d/2]] : ccw_path([-w2,-d2+r], [-w2+r,-d2], po=[-w2+r,-d2+r]),
    half ? [[w/2,-d/2]] : ccw_path([w2-r,-d2], [w2,-d2+r], po=[w/2-r,-d/2+r])
  );

// ====================================================================
// unit parametric functions: [0,1] -> [0,1]
// ====================================================================

unity_guide = function(t) t;
poly_damp = function(t, d=3) (1-pow(1-2*t, d))/2;
poly_ease = function(t, d=3) let(u=t*2, v=u-2, s=d%2?1:-1) u<1 ? pow(u,d)/2 : s*(pow(v,d)+s*2)/2;
line_guide = function(t, start=1, end) start+(ifundef(end, 1-start)-start)*t;
ripple_guide = function(t, d=5) (sin(360*t)+pow(sin(d*180*t),2))/4+0.5;
pulse_guide = function(t) exp(-5*pow(6*t-1,2)/pow(4+2*t,2));
round_guide = function(t) sqrt(0.251-pow(t-0.5,2))+0.5;
dome_guide = function(t) sqrt(1.001-t*t);
wave_guide = function(t, d=4) (cos(180*d*t)+3)/4;
fillet_guide = function(t, r0=0.1, r1) let(r1=ifundef(r1,r0)) // r0=head rounding, r1=tail rounding
  r0!=0 && t<abs(r0) ? (1-r0)+r0*sqrt(1-pow(1-t/abs(r0),2)) :
  r1!=0 && t>1-abs(r1) ? (1-r1)+r1*sqrt(1-pow(1-((1-min(1,t))/abs(r1)),2)) : 1;
scale_guide = function(t, r0=0.1, r1) let(r1=ifundef(r1,r0)) // r0=head rounding, r1=tail rounding
  r0!=0 && t<abs(r0) ? sqrt(1-pow(1-t/abs(r0),2)) :
  r1!=0 && t>1-abs(r1) ? sqrt(1-pow(1-((1-min(1,t))/abs(r1)),2)) : 1;
arc_guide = function(t, f=0.1) let(r=(f*f+0.25)/f/2, k=sqrt(r*r-0.25), s=asin(0.5/r), a=t*2*s-s) r*cos(a)-k;

// ====================================================================
// unit trace functions: [0,1] -> [0,1]^2 or [0,1]^3
// ====================================================================

apple_trace = function(t) let(a=0.67, b=1.1, p=0.15, q=0.08, u=360*t-90, r=a*(1-sin(u)), d=PI*(u/180-1/2), s=d*d) [-b*r*exp(-q*s)*sin(u)-a*0.4,r*exp(-p*s)*cos(u)];
butterfly_trace = function(t) let(u=360*t) 0.32*(abs(sin(2*u)) + sin(u/2))*[cos(u),sin(u)] + [0.1,0];
egg_trace = function(t) let(u=360*t) [4*cos(u),3*sin(u+sin(u)*11)]/8;
ring_trace = function(t, a=0, origin=[0.5,0.5]) [cos(360*t+a)/2,sin(360*t+a)/2]+origin;
wave_trace = function(t, d=1, f=1) [t,cos(360*d*t)*f/2+0.5];
arch_trace = function(t) t<0.5 ? ring_trace(t) : box_trace(t-1/8);
box_trace = function(t, sz=[1,1], center=false) let(x=sz[0], y=sz[1], k=(t%1)*4, c=(center?sz/2:[0,0]))
  k<1 ? [x,y]+k*[-x,0]-c :
  k<2 ? [0,y]+(k-1)*[0,-y]-c :
  k<3 ? [0,0]+(k-2)*[x,0]-c :
  [x,0]+(k-3)*[0,y]-c ;
puzzle_trace = function(t) let(a=127, r=0.3125, x=floor(t), k=(t%1)*4)
  k>3 ? [x+1+sin(360-a+(k-3)*a)*r,(1-r)+cos(360-a+(k-3)*a)*r] :
  k>1 ? [x+0.5+cos(270-a+(k-1)*a)*r,r+sin(270-a+(k-1)*a)*r] :
  [x+sin(k*a)*r,(1-r)+cos(k*a)*r];
fillet_trace = function(t, r0=0.1, r1) let(r1=ifundef(r1,r0)) // r0=head rounding, r1=tail rounding
  r0!=0 && t<r0 ? [1,r0*2]-ring_trace(0.25+0.25*t/r0)*r0*2 :
  r1!=0 && t>1-r1 ? [1,1]-ring_trace(0.75-0.25*(1-t)/r1)*r1*2 : [1,t];
inf_trace = function(t, z=0) let(a=t*360) [cos(a), sin(a*2)/2, z*sin(a)/4]/(3-cos(a*2)); // 3D, z=z scale
arc_trace = function(t, f=0.1) let(r=(f*f+0.25)/f/2, k=sqrt(r*r-0.25), s=asin(0.5/r), a=t*2*s-s) [r*sin(a)+0.5,r*cos(a)-k]; // t spans angle
bridge_trace = function(t, f=0.1) let(r=(f*f+0.25)/f/2, x=t-0.5) [t,sqrt(r*r-x*x)+f-r]; // t spans x-axis

// ====================================================================
// 2D paths between points
// ====================================================================

function line_path(p, from=[0,0]) = [from,p];
function ccw_path(p1, p2, f=2, i=0, po) = let(m=(p1+p2)/2, ov=orth2d(p2-p1), u=unit(ov))
  let(o=m+(po?u*(po-m)*u:ov/f), r=norm(p1-o)) norm(p1-p2)<=2*$fs ? i==0 ? [p1,p2] : [p1] :
  let(pm=o-r*u) concat(ccw_path(p1, pm, f, i+1, o), ccw_path(pm, p2, f, i+1, o), i==0?[p2]:[]);
function cw_path(p1, p2, f=2, i=0, po) = let(m=(p1+p2)/2, ov=orth2d(p1-p2), u=unit(ov))
  let(o=m+(po?u*(po-m)*u:ov/f), r=norm(p1-o)) norm(p1-p2)<=2*$fs ? i==0 ? [p1,p2] : [p1] :
  let(pm=o-r*u) concat(cw_path(p1, pm, f, i+1, o), cw_path(pm, p2, f, i+1, o), i==0?[p2]:[]);
function sin_path(p1, p2, n=1, m=0.5) = let(v=p2-p1, ov=orth2d(v)*m) [for (t=quanta(n*90, end=1)) p1+t*v+sin(t*n*360)*ov];
function cos_path(p1, p2, n=1, m=0.5) = let(v=p2-p1, ov=orth2d(v)*m) [for (t=quanta(n*90, end=1)) p1+t*v+cos(t*n*360)*ov];
function peanut_path(p1, p2, w=10, f=1.2) = let(m=(p1+p2)/2, e=p1-p2, h=norm(e), c=e[0]/h, s=e[1]/h, v=h/2) [for (t=quanta(ceil(perimeter(v, w)/8/$fs)*2)) let(a=t*360) m + [cos(a)*(v+f*w/2),(2*sin(a)-sin(a)^5)*w/2] * [[c,s],[-s,c]]];
function airfoil_path(p1, p2, w=5, f=0.3, b, s) = let(u=unit([p1.y-p2.y,p2.x-p1.x]), a1=opt(f,0,0.3), a2=is_list(f)?opt(f,1):0, m=(p1+p2)/2+ifundef(b,[0,0]), p11=p1-a1*u, p12=p1+a1*u, p21=p2-a2*u, p22=p2+a2*u, pm1=m-w*u/2, pm2=m+w*u/2) smooth([p11,pm1,p21,p22,pm2,p12], div=s, loop=true); // f=tip shaping, b=bias, s=resolution

// ====================================================================
// 3D paths between points
// ====================================================================

function ruler_path(p1, p2, n) = let(d=p2-p1) [for (t=quanta(n?n:ceil(norm(d)/$fs), end=1)) p1+t*d]; // 2D also works
function exp_path(p1, p2, n, r=2, v) = r==0 ? [p1,p2] : r==1 ? ruler_path(p1, p2, n) : let(v=(v!=undef?v:(p2-p1)*(1-r)/(1-pow(r,n)))) n==0 ? [p1] : let(p=p2-pow(r,n-1)*v) concat(exp_path(p1, p, n-1, r, v), [p2]); // divide into n segments of exponential ratios
function bush_path(p1, p2, f=3, end=1) = let(dx=p2[0]-p1[0], dy=p2[1]-p1[1], dz=p2[2]-p1[2]) [for (t=quanta(norm(p1-p2)/$fs, end=end)) let(s=poly_ease(t,f)) p1+[dx*s,dy*s,dz*t]];
function vault_path(p1, p2, n, loop=false) = let(q1=p1[2]<p2[2]?p1:p2, q2=p1[2]<p2[2]?p2:p1, m=q1+[0,0,q2[2]-q1[2]]) concat([for (t=quanta(ceil(n!=undef?n:norm(p2-p1)/$fs/2), end=1)) bezier([p1,m,m,p2], t)], loop?[m]:[]);
function helix_path(p1, p2, d=3, pitch=1) = pitch==0 ? [] : let(v=p2-p1, r=d/2, n=norm(v), m=m3_rotate(v)) [for (t=quanta(ceil(PI*r*n/pitch/$fs), end=1)) let(a=360*t*n/pitch) p1+[cos(a)*r,sin(a)*r,n*t]*m];

// ====================================================================
// 3D functions
// ====================================================================

// a line segment
function line(x, y, z) = [[0,0,if(z)0],[x,y,if(z)z]];

// an arbitrary orthogonal vector of v
function orth(v) = let(x=v[0], y=v[1], z=v[2]) x!=0 && y!=0 ? [y,-x,0] : x!=0 && z!=0 ? [z,0,-x] : y!=0 && z!=0 ? [0,z,-y] : [z, x, y];

// an orthogonal vector of v wrt point u
function orth2(v, u) = v-(u*v)/(v*v)*u;

// project vector v onto vector u
function proj(v, u) = (u*v)/(v*v)*u;

// project vector v onto a plane at origin defined by normal unit vector n
function proj2(v, n) = v-(v*n)*n;

// project vector v along direction d onto a plane at origin defined by normal vector n of any magnitude
function proj3(v, d, n) = v-(v*n)/(d*n)*d;

// project 3D points onto xy-plane wrt eye location [0,0,e]
function proj3d(points, e) = [for (p=points) [p[0],p[1]]*e/(e-p[2])];

// swap coordinates
function swap_xy(points) = [for (p=points) p[2]==undef ? [p[1],p[0]] : [p[1],p[0],p[2]]];
function swap_yz(points) = [for (p=points) [p[0],ifundef(p[2],0),p[1]]];
function swap_xz(points) = [for (p=points) [ifundef(p[2],0),p[1],p[0]]];

// check for a valid 3D point
function valid3d(point) = is_num(point[0]) && is_num(point[1]) && is_num(point[2]);

// ascend 2D or 3D points, dz=z increment {see force3d()}
function ascend3d(points, dz=0) = [for (p=points) [is_list(p)?p[0]:p,ifundef(p[1],0),ifundef(p[2],0)+dz]];

// a random list of 3D points
function random3d(n, min=-1, max=1, seed) = let(s=rnd_seed(seed)) [for (i=[1:n]) rnd(min, max, 3, i+s/i)];

// a random walk
function wander3d(n, min=-1, max=1, seed, trail=[[0,0,0]], base=[0,0,0]) = n>0 ? let(s=rnd_seed(seed)) wander3d(n-1, min, max, s, concat(trail, [trail[len(trail)-1]+rnd(min, max, 3, n+s/n)+base]), base) : trail;

// angle between 2 vectors
function angle3d(u, v=[0,0,1]) = atan2(norm(cross(v,u)), u*v);

// signed angle from vector u to v in reference to a plane with normal vector n
function s_angle3d(u, v, n) = atan2(cross(v,u)*n, u*v);

// calculate XYZ-convention Euler angles (may cause gimbal lock) from directional vector, usage: m4_euler(euler(v))
function euler(v) = let(m=m3_rotate(v)) [atan2(m[1][2], m[2][2]), asin(-m[0][2]), atan2(m[0][1], m[0][0])];

// return 1 or -1 if colinear, zero otherwise, t=threshold
function colinear(u, v, t=0.9999) = let(d=(u*v)/norm(u)/norm(v)) abs(d)>t ? sign(d) : 0;

// mash points to flatten the population (l=low, h=high)
function mash3d(points, l, h) = [for (p=points) let(z=l==undef?p[2]:max(l,p[2])) [p[0],p[1],h?min(h,z):z]];

// remove points outside of z range [l,h]
function filter3d(points, l, h) = [for (p=points) if (within(p[2], l, h)) p];

// scale 3D points, ratios=[xscale,yscale,zscale]
function scale3d(points, ratios, origin=[0,0,0]) = let(xs=opt(ratios,0), ys=opt(ratios,1,1), zs=opt(ratios,2,1)) [for (p=shift3d(points, -origin)) [p[0]*xs,p[1]*ys,ifundef(p[2],0)*zs]+origin];

// project 3D points onto the unit sphere (radius = 1) at origin
function unit3d(points, origin=[0,0,0]) = [for (p=points) p / norm(p-origin)];

// translate 2D or 3D points so that point i becomes the origin
function anchor(points, i=0) = let(k=len(points), i=(i+k)%k) [for (p=points) p-points[i]];

// adjust dimension(s)
function adjust3d(points, delta=[0,0,0]) = rank(points)<2 ? points+delta : [for (p=points) p+delta];

// shift 3D points
function shift3d(points, vector) = [for (p=points) [p[0]+vector[0],p[1]+vector[1],ifundef(p[2],0)+vector[2]]];

// shift 3D points as a whole to be just above xy-plane
function float3d(points, z=0) = let(k=-min(slice(points, 2))) shift3d(points, [0,0,k+z]);

// shift 3D points to be flush against an axis (determined by the sign of xsign, ysign and zsign)
// xsign, ysign, zsign: undef=no change, 0=center, -ve/+ve=which side to go
function flush3d(points, xsign, ysign, zsign, origin=[0,0,0]) =
  let(xx=minmax(slice(points, 0)), yy=minmax(slice(points, 1)), zz=minmax(slice(points, 2)))
  let(dx=origin[0] - (xsign == undef ? 0 : xsign == 0 ? avg(xx) : xsign > 0 ? xx[0] : xx[1]))
  let(dy=origin[1] - (ysign == undef ? 0 : ysign == 0 ? avg(yy) : ysign > 0 ? yy[0] : yy[1]))
  let(dz=origin[2] - (zsign == undef ? 0 : zsign == 0 ? avg(zz) : zsign > 0 ? zz[0] : zz[1]))
  shift3d(points, [dx,dy,dz]);

// centroid of 3D points
function centroid3d(points) = [for (i=[0:2]) avg(slice(points, i))];

// center 3D points
function center3d(points, at=[0,0,0]) = shift3d(points, at-centroid3d(points));

// revolve 3D points about z-axis
function spin3d(points, a) = points * m3_spin(a);

// compute collective normal for a loop
function cross3d(points, i=0, u) = let(k=len(points), u=ifundef(u, points[1]-points[0])) i<k-2 ? let(v=points[(i+2)%k]-points[0]) cross(u, v) + cross3d(points, i+1, v) : points[0]*0;

// convert points to 3D, set/override z if provided, otherwise preserve z or default to 0 {see ascend3d()}
function force3d(points, z) = rank(points)>1 ? [for (p=points) [p[0],p[1],z==undef?ifundef(p[2],0):z]] : len(points)>0 ? [points[0],points[1],z==undef?ifundef(points[2],0):z] : [];

// shake up 3D points
function shake3d(points, max=10, seed) = let(s=rnd_seed(seed)) [for (i=[0:len(points)-1]) points[i] + rnd(-max/2, max/2, 3, i+s/(i+1))];

// invert layers along z-axis
function invert3d(layers, h) = [for (p=layers) [for (i=[len(p)-1:-1:0]) [p[i][0],p[i][1],h-p[i][2]]]];

// reflect points vertically across xy-plane
function reflect3d(points) = [for (p=points) [p[0],p[1],-p[2]]];

// orient 3D points on xy-plane (normal=[0,0,1]) to new normal vector n, with uniform rotation a about n
function orient3d(points, n, from=[0,0,1], a=0) = points * m3_spin(a) * m3_rotate(n, from);

// wrap points on xy-plane around a vertical cylinder of diameter d, spanning angle a, reference width w
function polar3d(points, d=20, a=360, w) = let(w=ifundef(w, box2dw(points)), r=d/2) [for (p=points) let(t=p[0]*a/w, e=ifundef(p[2],0)) [cos(t)*(r+e), sin(t)*(r+e), p[1]*(d*PI*a)/(w*360)]];

// scale 2D points onto the surface of a sphere linearly with respect to origin
function land(points, r) = [for (p=points) unit(p)*r];

// project 2D points to the top surface of a sphere, f=scaling factor
function on_sphere(points, r, f=12) = [for (p=points) let(h=(90+p[0]*f),v=p[1]*f,q=(r+p[2])*[-cos(h)*cos(v), sin(h)*sin(v), sin(h)*cos(v)]) q*(r+p[2])/norm(q)];

// find center of a sphere given 3 points on its surface and radius r (negate r to get the other solution if available)
function sphere3p(p1, p2, p3, r) = let(p21=p2-p1, p31=p3-p1, n=cross(p21,p31), o=p1+cross((p21*p21)*p31-(p31*p31)*p21, n)/(n*n)/2, t=sign(r)*sqrt((r*r-(o-p1)*(o-p1))/(n*n))) t==t ? o+t*n : undef;

// return points on a grid of dimension dm=[x,y,z], divided evenly by ns=[nx,ny,nz] sections on each axes.
function grid3d(dm, ns=[2,2,2]) = [for (i=quanta(ns[0], end=1), j=quanta(ns[1], end=1), k=quanta(ns[2], end=1)) [i*dm[0]-dm[0]/2,j*dm[1]-dm[1]/2,k*dm[2]-dm[2]/2]]; 

// return n points "evenly"-distributed on a sphere of radius 1
function orbit(n) = let(c=180*(1+sqrt(5))) [for (i=[0.5:n-0.5]) let(h=acos(1-2*i/n), t=c*i) [cos(t)*sin(h), sin(t)*sin(h), cos(h)]];

// Fibonacci pattern of n points in a circle of radius 1
function sunflower(n) = let(c=180*(1+sqrt(5))) [for (i=[0.5:n-0.5]) let(r=sqrt(i/n), t=c*i) [r*cos(t), r*sin(t)]];

// vertices of a tetrahedron enclosed by a sphere of diameter d
function tetra(d=10) = let(f=d/(2*sqrt(3))) [for (i=[1,-1], j=[1,-1]) f*[i,j,i*j]];
function tetra_faces() = [[0,2,1],[0,1,3],[0,3,2],[1,2,3]];

// vertices of an icosahedron enclosed by a sphere of diameter d
function icosa(d=10) = let(phi=golden(), u=[-1,1], v=1/[phi,-phi]) [for (k=[0:2],j=u,i=v) [[i,j,0],[j,0,i],[0,i,j]][k]]*d/sqrt(1+2/(3+sqrt(5)))/2;
function icosa_faces() = [[0,9,1],[0,1,11],[0,6,7],[0,11,6],[0,7,9],[1,5,4],[1,4,11],[1,9,5],[2,3,8],[2,10,3],
  [2,7,6],[2,6,10],[2,8,7],[3,4,5],[3,10,4],[3,5,8],[4,10,11],[5,9,8],[6,11,10],[7,8,9]];

// vertices of an dodecahedron enclosed by a sphere of diameter d
function dodeca(d=10) = let(phi=golden(), u=[-1,1], v=1/[phi,-phi]) concat([for (i=u,j=u,k=u) [i,j,k]], [for (k=[0:2],j=v,i=v) [[0,i,1/j],[i,1/j,0],[1/j,0,i]][k]])*d/sqrt(3)/2;
function dodeca_faces() = [[0,11,10,2,19],[0,15,14,4,11],[0,19,18,1,15],[1,18,3,8,9],[2,10,6,12,13],[2,13,3,18,19],[3,13,12,7,8],[4,14,5,16,17],[4,17,6,10,11],[5,9,8,7,16],[5,14,15,1,9],[6,17,16,7,12]];

// Hilbert cube of size s with points at least u length apart (watch out for stack overflow)
function hilbert3(s, u=1, v=[0,0,0], d1=[1,0,0], d2=[0,1,0], d3=[0,0,1]) = (u==0 || s<=u) ? [v] : let(s=s/2, v=v
    - s*[min(d1[0],0),min(d1[1],0),min(d1[2],0)]
    - s*[min(d2[0],0),min(d2[1],0),min(d2[2],0)]
    - s*[min(d3[0],0),min(d3[1],0),min(d3[2],0)])
  concat(
    hilbert3(s, u, v, d2, d3, d1),
    hilbert3(s, u, v+s*d1, d3, d1, d2),
    hilbert3(s, u, v+s*(d1+d2), d3, d1, d2),
    hilbert3(s, u, v+s*d2, -d1, -d2, d3),
    hilbert3(s, u, v+s*(d2+d3), -d1, -d2, d3),
    hilbert3(s, u, v+s*(d1+d2+d3), -d3, d1, -d2),
    hilbert3(s, u, v+s*(d1+d3), -d3, d1, -d2),
    hilbert3(s, u, v+s*d3, d2, -d3, -d1)
  );

// ====================================================================
// matrix operations (rotations are counter-clockwise)
// ====================================================================

function m2_scale(s=[1,1]) = [[s[0],0],[0,s[1]]];

function m2_rotate(d) = [[cos(d),sin(d)],[-sin(d),cos(d)]];

function m3_xprod(v) = [[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]];

function m3_tensor(u, v) = [[u[0]*v[0],u[0]*v[1],u[0]*v[2]], [u[1]*v[0],u[1]*v[1],u[1]*v[2]], [u[2]*v[0],u[2]*v[1],u[2]*v[2]]];

// various rotations using a 3x3 matrix (left to right order), e.g. [[1,2,3]]*m3_spin(45) or points*m1*m2*m3

function m3_ident(k=1) = [[k,0,0],[0,k,0],[0,0,k]];
function m3_revolve(a, n) = let(u=n/norm(n)) m3_ident(cos(a)) + m3_xprod(u)*sin(a) + m3_tensor(u, u)*(1-cos(a));
function m3_spin(a) = [[cos(a),sin(a),0],[-sin(a),cos(a),0],[0,0,1]]; // around z-axis
function m3_negate(b) = let(b=ifundef(b, [0,0,1]), f=orth(b), v=cross(f, b)) m3_rotate(v, f, b)*m3_rotate(-f, v, b);
function m3_rotate(v, from=[0,0,1], basis) = let(s=colinear(v, from)) s<0 ? m3_ident(s) :
let(c=(basis!=undef?basis:cross(from, v)), q=append(c, from*v), r=unit(override(q, 3, q[3]+norm(q)))) [
  [1-2*r[1]*r[1]-2*r[2]*r[2],   2*r[0]*r[1]+2*r[2]*r[3],   2*r[0]*r[2]-2*r[1]*r[3]],
  [  2*r[0]*r[1]-2*r[2]*r[3], 1-2*r[0]*r[0]-2*r[2]*r[2],   2*r[1]*r[2]+2*r[0]*r[3]],
  [  2*r[0]*r[2]+2*r[1]*r[3],   2*r[1]*r[2]-2*r[0]*r[3], 1-2*r[0]*r[0]-2*r[1]*r[1]]];

// affine transformations (left to right order), e.g. [4,5,6,1]*m4_euler([10,20,30]) or m4_transform(points, m1*m2*m3)

function m4_transform(points, m4) = force3d([for (p=points) [p[0],p[1],ifundef(p[2],0),1] * m4]);
function m4_ident(k=1) = [[k,0,0,0], [0,k,0,0], [0,0,k,0], [0,0,0,1]];
function m4_scale(s=[1,1,1]) = [[s[0],0,0,0], [0,s[1],0,0], [0,0,s[2],0], [0,0,0,1]];
function m4_translate(v=[0,0,0]) = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [v[0],v[1],v[2],1]];
function m4_roll(a) = [[1,0,0,0], [0,cos(a),sin(a),0], [0,-sin(a),cos(a),0], [0,0,0,1]]; // around x-axis
function m4_pitch(a) = [[cos(a),0,-sin(a),0], [0,1,0,0], [sin(a),0,cos(a),0], [0,0,0,1]]; // around y-axis
function m4_spin(a) = [[cos(a),sin(a),0,0], [-sin(a),cos(a),0,0], [0,0,1,0], [0,0,0,1]]; // around z-axis
function m4_rotate(v, from=[0,0,1], basis) = let(s=colinear(v, from)) s!=0 ? mm_ident(s) :
  let(c=(basis!=undef?basis:cross(from, v)), q=append(c, from*v), r=unit(override(q, 3, q[3]+norm(q)))) [
  [1-2*r[1]*r[1]-2*r[2]*r[2],   2*r[0]*r[1]+2*r[2]*r[3],   2*r[0]*r[2]-2*r[1]*r[3], 0],
  [  2*r[0]*r[1]-2*r[2]*r[3], 1-2*r[0]*r[0]-2*r[2]*r[2],   2*r[1]*r[2]+2*r[0]*r[3], 0],
  [  2*r[0]*r[2]+2*r[1]*r[3],   2*r[1]*r[2]-2*r[0]*r[3], 1-2*r[0]*r[0]-2*r[1]*r[1], 0],
  [                        0,                         0,                         0, 1]];
function m4_euler(a=[0,0,0]) = [ // Euler angles suffer from gimbal lock, should use vector-based rotations instead
         [1, 0, 0, 0],
         [0, cos(a[0]), sin(a[0]), 0],
         [0, -sin(a[0]), cos(a[0]), 0],
         [0, 0, 0, 1]] // step 1: rotate a[0] degrees about x
         * [[cos(a[1]), 0, -sin(a[1]), 0],
         [0, 1, 0, 0],
         [sin(a[1]), 0, cos(a[1]), 0],
         [0, 0, 0, 1]] // step 2: rotate a[1] degrees about y
         * [[cos(a[2]), sin(a[2]), 0, 0],
         [-sin(a[2]), cos(a[2]), 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]]; // step 3: rotate a[2] degrees about z

// object transformations using multmatrix() (right to left order), e.g. multmatrix(m3*m2*m1) children();

function mm_ident(k=1) = [[k,0,0,0], [0,k,0,0], [0,0,k,0], [0,0,0,1]];
function mm_scale(s=[1,1,1]) = [[s[0],0,0,0], [0,s[1],0,0], [0,0,s[2],0], [0,0,0,1]];
function mm_translate(v=[0,0,0]) = [[1,0,0,v[0]], [0,1,0,v[1]], [0,0,1,v[2]], [0,0,0,1]];
function mm_roll(a) = [[1,0,0,0], [0,cos(a),-sin(a),0], [0,sin(a),cos(a),0], [0,0,0,1]]; // around x-axis
function mm_pitch(a) = [[cos(a),0,sin(a),0], [0,1,0,0], [-sin(a),0,cos(a),0], [0,0,0,1]]; // around y-axis
function mm_spin(a) = [[cos(a),-sin(a),0,0], [sin(a),cos(a),0,0], [0,0,1,0], [0,0,0,1]]; // around z-axis (yaw)
function mm_negate(b) = let(b=ifundef(b, [0,0,1]), f=orth(b), v=cross(f, b)) mm_rotate(v, f, b)*mm_rotate(-f, v, b);
function mm_reframe(x, y, z) = [[x[0],y[0],z[0],0], [x[1],y[1],z[1], 0], [x[2],y[2],z[2],0], [0,0,0,1]]; // map to axes
function mm_rotate(v, from=[0,0,1], basis) = let(s=colinear(v, from)) s!=0 ? mm_ident(s) :
  let(c=(basis!=undef?basis:cross(from, v)), q=append(c, from*v), r=unit(override(q, 3, q[3]+norm(q)))) [
  [1-2*r[1]*r[1]-2*r[2]*r[2],   2*r[0]*r[1]-2*r[2]*r[3],   2*r[0]*r[2]+2*r[1]*r[3], 0],
  [  2*r[0]*r[1]+2*r[2]*r[3], 1-2*r[0]*r[0]-2*r[2]*r[2],   2*r[1]*r[2]-2*r[0]*r[3], 0],
  [  2*r[0]*r[2]-2*r[1]*r[3],   2*r[1]*r[2]+2*r[0]*r[3], 1-2*r[0]*r[0]-2*r[1]*r[1], 0],
  [                        0,                         0,                         0, 1]];

// ====================================================================
// quaternion functions where q = [w,x,y,z], using qvq' convention (right to left order)
// e.g. q_orient(points, q_mult(q_map(u2, v2), q_map(u1, v1))) will apply u1->v1, then u2->v2
// ====================================================================

// intrinsic operators
function q_conj(q) = [q[0],-q[1],-q[2],-q[3]]; // conjugate
function q_inv(q) = unit(q_conj(q)); // inverse
function q_mult(q, r) = [ // multiplication
  q[0]*r[0]-q[1]*r[1]-q[2]*r[2]-q[3]*r[3],
  q[0]*r[1]+q[1]*r[0]+q[2]*r[3]-q[3]*r[2],
  q[0]*r[2]-q[1]*r[3]+q[2]*r[0]+q[3]*r[1],
  q[0]*r[3]+q[1]*r[2]-q[2]*r[1]+q[3]*r[0]];

// identity
function q_ident() = [1,0,0,0];

// quaternion causing a rotation of angle a about axis n
function q_revolve(a, n=[0,0,1]) = let(a2=a/2, n=unit(redim(n, 3))) prepend(n*sin(a2), cos(a2));

// quaternion that maps u to v, with optional n, the axis of rotation when u and v are antiparallel (if n is undefined, an arbitrary axis is picked which may be fine for a single vector but could cause discontinuity in a coherent set)
function q_map(u, v, n) = let(c=colinear(u, v)) c>0 ? q_ident() : c<0 ? let(n=(n!=undef?n:orth(u))) prepend(unit(n), 0) : let(w=cross(u, v), q=prepend(w, u*v)) unit(plus(q, 0, norm(q)));

// rotate one point
function q_rotate(point, q) = let(r=[0,point[0],point[1],point[2]], s=q_mult(q_mult(q, r), q_conj(q))) [s[1],s[2],s[3]];

// rotate a set of points
function q_orient(points, q) = [for (p=points) q_rotate(p, q)];

// utility to convert quaternion q to Euler angles (XYZ convention), may lead to gimbal lock issues near ±90º
function q_euler(q) = [
  atan2(2*(q[0]*q[1]+q[2]*q[3]), 1-2*(q[1]*q[1]+q[2]*q[2])),
  asin(2*(q[0]*q[2]-q[1]*q[3])),
  atan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]*q[2]+q[3]*q[3]))];

// ====================================================================
// layer manipulation functions
// ====================================================================

// shift layers
function shift_layers(layers, delta=[0,0,0], base=false) = let(m=[for (l=layers) shift3d(l, delta)]) base ? prepend(m, force3d(m[0], 0)) : m;

// scale layers based on a unit guiding path representing [layer, scale]
function scale_layers(layers, guide) = let(k=len(layers)-1) [for (i=[0:k]) [let(s=lookup(i/k, guide)) for (p=layers[i]) [p[0]*s, p[1]*s, p[2]]]];

// spin layers about z-axis based on a unit guiding path representing [layer, angle]
function spin_layers(layers, guide, a=60) = let(k=len(layers)-1, g=ifundef(guide, [[0,0],[1,a]])) [for (i=[0:k]) spin3d(layers[i], lookup(i/k, g))];

// make sure layers can be rendered by layered_block()
function rectify_layers(layers, invert=false) = let(k=len(layers)) k<2 || len(layers[0])<2 ? layers :
  let(a=layers[0], b=layers[1], u=a[1]-a[0], v=a[len(a)-1]-a[0], w=b[0]-a[0])
  reverse(layers, enable=xor(sign3v(u, v, w)<0, invert));

// convert layers to slope lines and vice versa, d=dilution, s=twist
function isomesh(layers, d=1, s=0) = let(n=len(layers[0])) [for (i=[0:d:n-1]) [for (j=[0:len(layers)-1]) layers[j][round(i+n+j*s)%n]]];

// ====================================================================
// morph functions
// ====================================================================

// morph linearly between 2 profiles (2D or 3D) of same cardinal at t between [0,1], f=scaling factor
function morph(p1, p2, t, f=1) = p1==p2 && f==1 ? p1 : [for (i=[0:len(p1)-1]) let(a=p1[i], b=p2[i]*f) a+(b-a)*t];

// interpolate layers by morphing between 2 profiles (2D or 3D) vertically, h=height, n=samples, s=cyclic offset
// optional scaler=[[t,f],...] where t=proportion in [0,1], f=scaling factor
// result is a set of layers ready for layered_block() <<deprecated in preference to morph_smooth()>>
function morph_between(p1, p2, h, scaler, n, s=0, z=0, end=1, curved=true) = 
  let(n=(n!=undef?n:max(len(p1),len(p2))), q1=cyclic(resample(p1, n), s), q2=resample(p2, n))
  scaler==undef ?
    [for (t=quanta(ceil(abs(h)/2/$fs), end=end)) ascend3d(morph(q1, q2, t, curved?t:1), t*h+z)] :
    [for (f=scaler) ascend3d(morph(q1, q2, f[0])*f[1], f[0]*h+z)];

// interpolate layers by morphing along a series of profiles (2D or 3D) vertically with given intervals
// result is a set of layers ready for layered_block() <<deprecated in preference to morph_smooth()>>
function morph_multiple(profiles, intervals, n, curved=false, z=0, i=0) =
  let(n=(n!=undef?n:max([for (p=profiles) len(p)])))
  i == len(profiles)-1 ? [ascend3d(resample(last(profiles), n), z)] : concat(
    morph_between(profiles[i], profiles[i+1], intervals[i], n=n, z=z, end=0.999, curved=opt(curved, i)),
    morph_multiple(profiles, intervals, n, curved, z+ifundef(intervals[i],0), i+1)
  );

// from profile create a cookie-shape mesh of height h, vertical base height b, peak at origin, f=obesity
function morph_cookie(profile, h, b=0, origin=[0,0], f=0) =
  let(p=shift2d(profile, -origin), n=ceil(perimeter(avg(size2d(profile))/4, h-b)/4/$fs))
  let(m=[for (t=quanta(n, end=1, max=90+f)) force3d(shift2d(p*cos(t-f), origin), b+(h-b)*sin(min(90,t)))])
  b<=0 ? m : prepend(m, force3d(shift2d(p, origin), 0));

// smooth vertical morphing for a series of profiles spaced with given intervals, see also fillet_sweep()
// n=resolution, f=smoothness, d=alignment_axis (see refit2d)
function morph_smooth(profiles, intervals, n=200, f=7, d=1) = let(h=accum(intervals), p=[for (i=indices(profiles)) force3d(refit2d(resample(profiles[i], n), d=d), h[i])]) isomesh([for (q=isomesh(p)) smooth(q, f, loop=false)]);

// ====================================================================
// 3D sweep functions - note that they return the layers in reverse order of path for correct surface norms
// sweep_profile - sweep a constant profile along path
// sweep_pipe - sweep a constant profile along path maintaining a consistent cross-section area
// sweep_wall - sweep a constant profile along path maintaining its vertical orientation
// sweep_layers - sweep a set of layers along path (no morphing)
// ====================================================================

function sweep_profile(profile, path, loop=false, s=0, i=0, a, m, t) = let(k=len(path)) i==k ? [] :
  let(profile=i==0 ? force3d(profile) : profile, a=i==0 ? m3_spin(sweep_twist(path, loop, k, s)) : a)
  let(tt=sweep_tangent(path, loop, k, i), mm=(i==0 ? sweep_init(tt) : m*m3_rotate(tt, t)))
  concat(sweep_profile(profile, path, loop, s, i+1, a, a*mm, tt), [shift3d(profile*mm, path[i])]);

// f in [0,1] determines how steep an angle has to be before toning it down by distortion (the less the steeper)
function sweep_pipe(profile, path, loop=false, s=0, f=0.3, i=0, a, m, t) = let(k=len(path)) i==k ? [] :
  let(c=i>0 ? profile : force3d(profile), a=i>0 ? a : m3_spin(sweep_twist(path, loop, k, s)))
  let(p0=path[(i+k-1)%k], p1=path[i%k], p2=path[(i+1)%k])
  let(v1=unit(loop ? p1-p0 : path[max(1,i)]-path[max(0,i-1)]))
  let(v2=unit(loop ? p2-p1 : path[min(k-1,i+1)]-path[min(k-2,i)]))
  let(tt=-v1==v2?v2:v1+v2, mm=i>0 ? m*m3_rotate(tt, t) : sweep_init(tt))
  let(d=unit(v2-v1)*(v1*v2<0 && norm(tt)<f?0.99:0))
  let(cp=[for (j=c*mm*m3_rotate(v1,tt)) let(q=proj3(j, v1, tt)) q-(q*d)*d+p1])
  concat(sweep_pipe(c, path, loop, s, f, i+1, a, a*mm, tt), [cp]);

function sweep_wall(profile, path, loop=false, s=0, tilt=true, i=0, m, t) = let(k=len(path)) i==k ? [] :
  let(profile=i==0 ? force3d(profile) : profile, ti=sweep_tangent(path, loop, k, i))
  let(tt=[ti[0],ti[1],0], mp=m3_rotate([0,-ti[2]/norm(tt),tilt?1:1e20]))
  let(mm=(i==0 ? sweep_init(tt) : m*m3_rotate(tt, t)))
  concat(sweep_wall(profile, path, loop, s, tilt, i+1, mm, tt), [shift3d(profile*mp*mm, path[i])]);

function sweep_layers(layers, path, loop=false, s=0, twist=true, i=0, a, m, t) = let(k=len(path)) i==k ? [] :
  let(a=i==0 ? twist ? m3_spin(sweep_twist(path, loop, k, -s)) : m3_ident() : a)
  let(c=layers[floor(len(layers)*i/k)], ti=sweep_tangent(path, loop, k, i), tt=twist ? ti : [ti[0],ti[1],0]) 
  let(mm=(i==0 ? sweep_init(tt) : m*m3_rotate(tt, t)), mc=twist ? mm : m3_rotate([0,-ti[2]/norm(tt),1])*mm)
  concat(sweep_layers(layers, path, loop, s, twist, i+1, a, a*mm, tt), [shift3d(force3d(c, 0)*mc, path[i])]);

// subroutine to find tangent at point i of path, k=length of path
function sweep_tangent(path, loop, k, i) = 
  let(u=loop ? path[i]-path[mod(i-1,k)] : path[max(1,i)]-path[max(0,i-1)])
  let(v=loop ? path[mod(i+1,k)]-path[i] : path[min(k-1,i+1)]-path[min(k-2,i)]) unit(u)+unit(v);

// subroutine to find the average twist along the path, k=length of path, s=additional twist (s*360º)
function sweep_twist(path, loop, k, s=0, i=0, m, t=[0,0,1]) = loop==false ? s*360/k :
  i==k ? let(t0=unit(sweep_tangent(path, loop, k, 0)), v0=orth(t0)) (s*360-s_angle3d(v0, v0*m, t0)%180)/k :
  let(tt=sweep_tangent(path, loop, k, i))
  sweep_twist(path, loop, k, s, i+1, i==0 ? m3_ident() : m*m3_rotate(tt, t), tt);

// subroutine to determine the initial rotation
function sweep_init(t) = m3_rotate(t[0]==0 && t[1]<0 && t[2]==0 ? [0,1,0] : [0,-1,0])*m3_rotate(t, [0,1,0]);

// ====================================================================
// 2D areas
// ====================================================================

// 2D area enclosing a profile, t=thickness, rounded=ends rounded, inflate=faster way to thicken line but it always produce rounded ends
module line2d(profile, t=0.003, rounded=false, loop=false, inflate=0) {
  if (len(profile)>1) offset(inflate) polygon(fence2d(profile, t, rounded=rounded, loop=loop, tidy=t>0.003?undef:0), convexity=9);
}

// 2D area of a grid where w=x-size, d=y-size, s=spacing, t=thickness of lines
module grid2d(w, d, s=10, t=1, center=true) {
  s0 = center ? 0 : s/2;
  for (i=[s0:s:w/2]) {
    translate([i,0]) square([t,d+t], center=true);
    if (i != 0) translate([-i,0]) square([t,d+t], center=true);
  }
  for (j=[s0:s:d/2]) {
    translate([0,j]) square([w+t,t], center=true);
    if (j != 0) translate([0,-j]) square([w+t,t], center=true);
  }
}

// 2D negative area of a random voronoi: d=diameter, t=line thickness, r=rounding, f=zoom factor
// a 2D child, if provided, will clip its boundry
module voronoi(d, t=1, r=1, f=0.6, seed) {
  seed=(seed!=undef?seed:ceil(rands(0, 10000, 1)[0]));
  if ($debug) echo(seed=seed);
  n = ceil(pow(d, 1.5)/10);
  e = rands(-d*f, d*f, n*2, seed);
  points = [for (i=[0:n-1]) [e[i],e[i+n]]];
  intersection() {
    if ($children) children();
    for (p=points) offset(r) offset(-r) intersection_for (q=points) {
      if (p!=q) translate((p+q)/2 + unit(p-q)*(t/4+r/2))
        rotate([0,0,atan2(p[0]-q[0], q[1]-p[1])]) translate([-d,-d]) square([2*d,d]);
    }
  }
}

// 2D negative area of a set of circles (holes)
module punch2d(locs=[[0,0]], m=3) {
  difference() {
    children();
    for (p=locs) translate(p) circle(d=m);
  }
}

// ====================================================================
// 3D objects
// ====================================================================

// a tiny, invisible dot (tetrahedron)
module dot(p=[0,0,0], r=0.001) { translate(p) polyhedron(tetra(r), tetra_faces(), convexity=10); }

// a disc
module disc(d=10, h=1) { solid(ring_path(d), h); }

// a dome (hemisphere), d=diameter, t=thickness (may be negative, zero means filled)
module dome(d=10, t=0, a=[0,90]) {
  c = t>0 ? confine(d-t*2, 0, d) : -t*2;
  a = [confine(a[0], -90, a[1]), confine(a[1], a[0], 90)];
  rotate_extrude(convexity=9) difference() {
    polygon(append(ring_path(d, a=a), [0,0]));
    if (c>0 && c<d) polygon(append(ring_path(c, a=a), [0,0]));
  }
}

// a cube sitting on xy-plane
module slab(dm=[1,1,1]) {
  translate([0,0,dm[2]/2]) cube(dm, center=true);
}

// a rectangular pad with rounded corners
module pad(x, y, h=1, bottom=0, r=1) {
  solid(pad_path(w=x, d=y, r=r), h=h, bottom=bottom);
}

// a right-angle vault between 2 points for ceiling support, t=thickness, n=resolution
module vault(p1, p2, t=1, n) {
  c = vault_path(p1, p2, n, loop=true);
  m = last(c);
  u = unit(cross(m-p1, m-p2));
  layered_block([shift3d(c, -u*t/2), shift3d(c, u*t/2)]);
}

// a pyramind from a profile, origin=center of offset if provided (defaults to profile center)
module pyramind(profile, h=5, inset=3, scale=0, origin) {
  c = (origin!=undef?origin:box2dc(profile));
  p = origin ? shift2d(profile, c) : profile;
  if (scale!=0)
    translate(c) deepen(h, scale=scale) polygon(shift2d(profile, -c));
  else if (inset!=0)
    layered_block(reverse([force3d(profile), force3d(offset2d(p, -inset), h)], enable=h<0));
}

// trace a thread along profile (negatively if children exists), r=tapering ratio (snap to points)
module trace(profile, d=0.2, r=0, loop=false, fuse=true) {
  p = force3d(fuse ? fuse(profile, loop=loop) : profile);
  s = r>0 ? [for (t=quanta(len(profile)-1, end=1)) [t,scale_guide(t, r)]] : undef;
  difference() {
    if ($children) children();
    sweep(ring_path(d), path=p, scaler=s, loop=loop);
  }
}

// a pipeline along profile, maintaining a cross-section of a proper circle as much as possible, unlike trace()
module pipeline(profile, d=1, loop=false) {
  difference() {
    if ($children) children();
    layered_block(sweep_pipe(ring_path(d), path=force3d(profile), loop=loop), loop=loop);
  }
}

// a plate created from a profile (list of v2 points)
// h=height, t=added thickness, bottom=ascend, inflate=added thickness, r=rounding
module plate(profile, h=1, t=0, bottom=0, inflate=0, r=2) {
  ascend(bottom) linear_extrude(h, convexity=9) offset(t + inflate) unsharp(r) polygon(profile, convexity=9);
}

// a shell enclosing the profile, negative t => inner shell
// h=height, t=thickness of wall (negative => inner wall), bottom=ascend, inflate=added thickness, r=rounding
module shell(profile, h=2, t=1, bottom=0, inflate=0, r=2) {
  if (t!=0) ascend(bottom) linear_extrude(h, convexity=9) difference() {
    offset(t>0 ? t : 0) offset(inflate) unsharp(r) polygon(profile, convexity=9);
    offset(t>0 ? 0 : t) offset(inflate) unsharp(r) polygon(profile, convexity=9);
  } 
}

// a wall based on profile, t could be negative (see also fillet_tray)
module wall(profile, h=20, t=1.6, flat=true) {
  if (t!=0 && h!=0) {
    tt = abs(t);
    hh = flat ? abs(h) : abs(h)-tt/2;
    c = flat ? [-tt,0] : ccw_path([tt,0], [0,0], po=[tt/2,0]);
    g = concat2d([[tt,0],[0,hh],c,[0,-hh]], [min(0,t),0]);
    layered_block([for (i=g) force3d(offset2d(profile, i[0]), i[1])], loop=true);
  }
}

// a twisted shell enclosing the profile
// h=height, t=thickness (negative => inner shell), m=hole size, pitch=height per turn (zero => disable twist)
function rotini(profile, h=20, t=0, m=0, pitch=100) = let(s=t>0?1:0, a=pitch==0?0:360/pitch) concat(
  [if (t!=0) let(p=offset2d(profile, t)) for (z=quanta(ceil(h/$fs), start=1-s, end=s, max=h)) force3d(spin2d(p, z*a), z)],
  [if (t==0 && m>0) let(p=[for (i=profile) unit(i)*m/2]) for (z=quanta(ceil(h/$fs), start=1-s, end=s, max=h)) force3d(spin2d(p, z*a), z)],
  [for (z=quanta(ceil(h/$fs), start=s, end=1-s, max=h)) force3d(spin2d(profile, z*a), z)]);
module rotini(profile, h=20, t=0, m=0, pitch=100) { layered_block(rotini(profile=profile, h=h, t=t, m=m, pitch=pitch), loop=(t!=0||m>0)); }

// a basin (combination of a plate and a shell)
// h=height, t=thickness of wall (negative => inner wall), bottom=ascend, inflate=added thickness, r=rounding
module basin(profile, h=5, t=1, bottom=0, inflate=0, r=2) {
  shell(profile, h=h, t=t, bottom=bottom, inflate=inflate, r=r);
  plate(profile, h=abs(t), t=0, bottom=bottom, inflate=inflate, r=r);
}

// a strip resting on its side along path, h=height, t=thickness, r=rounding, f=vertical fillet, s=path softening
module strip(path, h=10, t=1, r=0, f=0, s=0) {
  path = soften(path, s, loop=false);
  f = min(f, h/2-1);
  r = min(r, h/2-f);
  k = len(path);
  c0 = snip(cw_path([0,-1], [1,0], $fs=$fs/r), 1);
  c1 = snip(cw_path([0,0], [1,-1], $fs=$fs/r), 0, 1);
  pp = elong(path, -r, -r);
  e0 = pp[0]-path[0];
  e1 = path[k-1]-pp[k-1];
  m0 = r<0.5 ? [] : sweep_layers([for (i=c0) pad_path(t, h+i[1]*r*2, r=f)], [for (i=c0) as3d(path[0]+i[0]*e0)]);
  m1 = r<0.5 ? [] : sweep_layers([for (i=c1) pad_path(t, h+i[1]*r*2, r=f)], [for (i=c1) as3d(pp[k-1]+i[0]*e1)]);
  m = concat(m1, sweep_pipe(pad_path(t, h, r=f), force3d(r<0.5?path:pp)), m0);
  ascend(h/2) layered_block(m);
}

// a beam between 2 points (cross section is rectangular or circular)
// dm=cross section dimensions (2D for retangle, 1D for circle), c=cap length)
module beam(p1, p2, dm=0.5, c=0) {
  a = as3d(p1);
  b = as3d(p2);
  p = is_list(dm) ? quad_path(dm[0], dm[1]) : ring_path(dm);
  fillet_sweep(p, [a,b], c0=c, c1=c);
}

// an arrow of vector v at point p, d=diameter, c=cone length, r=cone-beam ratio (overrides c)
module arrow(v=[0,0,5], p=[0,0,0], d=0.25, c=2, r) {
  if (v != undef) {
    v = as3d(v);
    n = norm(v);
    q = confine(r!=undef ? 1-r : 1-c/n, 0, 1);
    beam(p, p+v*q, d);
    translate(p+v*q) orient(v) cylinder(n*(1-q), d, 0, $fn=_fn(d/2));
  }
}

// visualize a set of points (v2 or v3) as cubes
module mark(points, s=0.5, z=0, color="red", font=0) {
  p = ascend3d(has(points) ? has(points[0]) ? has(points[0][0]) ? points : [points] : undef : undef, z);
  if (p) {
    color(color) for (i=[0:len(p)-1]) translate(p[i]) {
      if (s>0) cube(s, center=true);
      if (font>0) translate([0,0,font]) flipx(90) label(str(i), ysize=font, h=0.1, center=true);
    }
  }
}

// plotting a profile (list of v2 or v3 points) as a 3D pipe, d=cross-section diameter, dot=enable dots,
// dup=debug overlaps, div=maximum segment size (avoids stack overflow; imperfections may arise where segments meet)
module plot(profile, d=0.2, loop=false, color="gold", dot=true, dup=false, div=1000) {
  p = force3d(fuse(profile, loop=loop));
  m = len(p);
  if (m>0) {
    if (m>1) color(color) {
      n = ceil(m/div); // break long profile into n segments
      if (n==1) layered_block(sweep_pipe(ring_path(d), path=p, loop=loop), loop=loop);
      else {
        alert(str("Plotting ", m, " points in ", n, " segment(s) starting from ", profile[0]));
        for (i=[0:n-1]) layered_block(sweep_pipe(ring_path(d), path=subarray(p, i*div, min(m-1,i*div+div))));
        if (loop) layered_block(sweep_pipe(ring_path(d), path=[p[m-1],p[0]]));
      }
    }
    if (dot && $preview) {
      for (i=[0:m-1]) translate(p[i]) {
        s = d + min(d*0.5,i==0?2:1);
        if (i>0 && i<m-1) %sphere(d=s, $fn=_fn(d));
        else color(i==0 ? "blue" : i==m-1 ? "tan" : undef) sphere(d=s, $fn=_fn(d));
      }
    }
    if (dup && $preview) { // for debugging
      for (i=seams(profile, true)) let(v=profile[i]) color(color) arrow([0,0,-3], [v[0],v[1],ifundef(v[2],0)+3.2]);
    }
  }
}

// alias for linear_extrude() with a default convexity (supports h<0 for extruding downwards)
module deepen(h, center, convexity, twist, slices, scale) {
  mirror([0,0,h<0?1:0]) linear_extrude(height=abs(h), center=center, convexity=ifundef(convexity,9), twist=twist, slices=slices, scale=scale) children();
}

// linear_extrude from a profile, h=height, t=thickness (if hollow, or 0 for filled), bottom=ascend
module solid(profile, h=1, t=0, bottom=0, scale=1, twist=0, inflate=0, loop=true) {
  if (h!=0) {
    p = force2d(profile);
    if (t==0) translate([0,0,bottom]) deepen(h, scale=scale, twist=twist) offset(inflate) polygon(p, convexity=9);
    else translate([0,0,bottom]) deepen(h, scale=scale, twist=twist) line2d(p, t+inflate, loop=loop);
  }
}

// plot a circle around profile (non-optimal)
module lasso(profile, s=0.2, loop=true, color="red") {
  e = encircle2d(profile);
  plot(shift2d(ring_path(e[1]*2), e[0]), d=s, loop=loop, color=color);
}

// given vertices v, produce a wireframe from edges under certain length (limit * shortest edge)
// range can be used to filter out points not in the array-index vicinity
module wire_hull(v, t=1, limit=1.05, range=0, smooth=true) {
  a = pairs(v, range);
  m = min([for (p=a) norm(p[0]-p[1])]) * limit;
  for (p=a) if (m==0 || norm(p[0]-p[1])<m) beam(p[0], p[1], t);
  if (smooth) clone_at(v) sphere(d=t);
}

// wheel-throwing a profile: t=thickness, shift=profile offset, a=angle range, spin=2d spin, vase=close bottom
module throw(profile, t=1, shift=[0,0], a=[0,360], spin=0, vase=false) {
  p = spin2d(profile, spin);
  shift = is_list(shift) ? shift : [shift,0];
  b = shift2d(box2d(p, t/2), shift);
  c = shift2d(p, shift);
  rotate([0,0,min(a)]) rotate_extrude(angle=abs(a[1]-a[0]), convexity=9) intersection() {
    line2d(vase ? prepend(c, [0,c[0][1]]) : c, inflate=t/2);
    polygon([[0,b[0][1]],b[1],b[2],[0,b[2][1]]], convexity=9);
  }
}

// create a lathe from a profile: shift=profile offset, a=angle range, spin=2d spin, inflate=thicken, fill=center-filled
module lathe(profile, shift=[0,0], a=[0,360], spin=0, inflate=0, fill=true) {
  p = spin2d(profile, spin);
  shift = is_list(shift) ? shift : [shift,0];
  b = shift2d(box2d(p, inflate), shift);
  c = shift2d(p, shift);
  rotate([0,0,min(a)]) rotate_extrude(angle=abs(a[1]-a[0]), convexity=9) intersection() {
    offset(inflate) polygon(fill ? concat([[0,c[0][1]]], c, [[0,c[len(c)-1][1]]]) :  c, convexity=9);
    polygon([[0,b[0][1]],b[1],b[2],[0,b[2][1]]], convexity=9);
  }
}

// a rounded tip with base diameter d, vertical spacing s, angle range a
module tip(d=5, s=0, a=[0,360], inflate=0) {
  r = d/2;
  p = concat2d([[[0,0],[r,0],[r,s]], arc_path(r*2, [180,150]), reverse(arc_path(r*2.63, [90,-30]))]);
  difference() {
    if ($children) children();
    lathe(p, a=a, inflate=inflate);
  }
}

// a vertical pipe at origin
// d=outer diameter, h=height, t=thickness of wall, m=diameter of hole (overrides t, zero for cylinder)
// floor=bottom of a closed pipe (default: none, negative: open pipe extended below ground)
module pipe(d, h=2, t=2, floor=0, center=false, flat=true, m) {
  r = d/2;
  t = m!=undef ? max(0.01,d-m)/2 : t>0 ? min(r,t) : max(0.01,r+t);
  a = center ? -h/2 : 0;
  if (abs(r-t)<0.01 && flat) {
    translate([0,0,a+min(0,floor)]) cylinder(d=d, h=h-min(0,floor));
  }
  else {
    g = flat ? h : h-t/2;
    c = (flat||t<$fs) ? [[0,0],[-t,0]] : arc_path(t);
    p = concat2d([[[0,min(0,floor)],[0,g]], c, [[-t,g],[-t,floor]]]);
    *plot(p);
    m = sweep_wall(p, force3d(ring_path(d), a), loop=true);
    layered_block(m, loop=true);
    if (floor>0) layered_block([slice(m, len(m[0])-1), slice(m, 0)]);
  }
}

// basic torus
module torus(d, w=1, a=[0,360]) {
  c = concat(round_path(w/2, w/2, a=a), abs(a[1]-a[0])<360 ? [[0,0]] : []);
  rotate_extrude(convexity=9) translate([d/2,0,0]) polygon(c, convexity=9);
}

// a lever arm with holes at locs, h=height, m=hole diameter, t=thickness
module lever(locs=[[-10,0],[10,0]], h=2, m=3, t=9, hole=true, curved=false) {
  k = len(locs);
  if (k>1) {
    p = curved && k>2 ? smooth(locs, loop=false) : locs;
//punch(hole ? locs : [], m=m) fillet_extrude(refine(soften(fence2d(p, t), (t-m)/2)), h=h);
    punch(hole ? locs : [], m=m, zz=[-0.01,h+0.01]) deepen(h) offset(t/2) line2d(p);
  }
}

// a spring ascending counterclockwisely with outer diameter m, height h, wire diameter w, starting angle a
module spring(m, pitch=5, h=10, w=2, a=0, clip=true) {
  r = m/2;
  n = h/pitch;
  e = w/(h*2); // extra length for clean clipping
  g = [for (t=quanta(ceil(_fn(r)*abs(n)), start=-e, end=1+e)) [r*cos(t*n*360+a),r*sin(t*n*360+a),h*t]];
  if (clip) clip_slice([-0.001,h+0.001], m+w) sweep(ring_path(w/2)*2, g);
  else {
    sweep(ring_path(w/2)*2, g);
    clone_at([g[0],last(g)]) sphere(d=w);
  }
}

// 3D text label at point p with height h, size limited by xsize and ysize, optional inflation and centering
module label(txt, p=[0,0,0], xsize=0, ysize=0, h=1, inflate=0, center=false, dir="ltr", font=$font, sf=0.5) {
  txt = is_list(txt) ? txt : [txt];
  n = len(txt);
  th = ysize / (n+(n-1)*sf); // text height
  rh = ysize==0 ? 15 : th * (1+sf); // row height
  for (i=[0:n-1]) translate(as3d(p)+[center?0:xsize/2,-rh*i+(center&&n>1?ysize/2:0),0])
    linear_extrude(h, convexity=10) offset(inflate) resize([xsize,th]*0.99, auto=true)
    text(txt[i], halign=center||xsize>0?"center":"left", valign=n==1?(center?"center":"bottom"):"top", direction=dir, font=font);
}

// a text-displaying signboard, ysize is auto if zero, d=text depth, passes $dm=[x,y,h,d,r,margin] to children
module signboard(txt, xsize=50, ysize=0, h=3, d=1, inflate=0.1, center=true, dir="ltr", font=$font, r, margin) {
  k = len(txt);
  m = margin!=undef ? margin : (k==0 ? 0 : k<3 ? xsize/5 : xsize/(k*2));
  x = xsize>0 ? max(0, xsize-m*2) : 50;
  y = ysize>0 ? max(0, ysize) : k<2 ? xsize : x*2/(k+1) + m*2;
  r = ifundef(r, y/10);
  translate(center ? [0,0] : [xsize,y]/2) difference() {
    f = min(h/2, r/2);
    fillet_extrude(pad_path(w=xsize, d=y, r=r), h=h, xz0=[0,0], xz1=[-f,-f]);
    if (k>0) label(txt, [0,0,h-d+0.01], xsize=(k<2?0:x), ysize=(k<2?x:0), h=d, inflate=inflate, center=true, dir=dir, font=font);
    if ($children>0) { $dm=[xsize,y,h,d,r,m]; children(); }
  }
}

module icosasphere(d, n=1, v, f) {
  function vsub(a, b, c, r) = let(d=(a+b)/2, e=(b+c)/2, f=(c+a)/2) [a, b, c, r*d/norm(d), r*e/norm(e), r*f/norm(f)];
  v = (v!=undef?v:icosa(d));
  f = (f!=undef?f:icosa_faces());
  if (n == 0) polyhedron(v, f, convexity=10);
  else {
    vv = [for (t=f) each vsub(v[t[0]], v[t[1]], v[t[2]], d/2)];
    ff = [for (i=[0:6:len(vv)-1]) each [[i,i+3,i+5],[i+1,i+4,i+3],[i+2,i+5,i+4],[i+3,i+4,i+5]]];
    rs = norm(vv[ff[0][0]] - vv[ff[0][1]]); // check face size
    icosasphere(d, rs<$fs?0:n-1, vv, ff);
  }
}

module icosabouquet(d, n=1, v, f) {
  v = (v!=undef?v:icosa(d));
  f = (f!=undef?f:icosa_faces());
  if (n == 0) polyhedron(v, f, convexity=10);
  else {
    r = d/2;
    s = len(v);
    vv = concat(v, [for (t=f) let(c=centroid3d([for (i=t) v[i]])) c*r/norm(c)]);
    ff = [for (i=incline(f)) for (j=[0:2]) [s+i, f[i][j], f[i][(j+1)%3]]];
    icosabouquet(d, n-1, vv, ff);
  }
}

module dodecasphere(d, n=1, v, f) {
  v = (v!=undef?v:dodeca(d));
  f = (f!=undef?f:dodeca_faces());
  if (n == 0) polyhedron(v, f, convexity=10);
  else {
    r = d/2;
    s = len(v);
    vv = concat(v, [for (t=f) let(c=centroid3d([for (i=t) v[i]])) c*r/norm(c)]);
    ff = [for (i=incline(f)) for (j=incline(f[i])) [s+i, f[i][j], f[i][(j+1)%5]]];
    icosasphere(d, n-1, vv, ff);
  }
}

module dodecabouquet(d, n=1, v, f) {
  v = (v!=undef?v:dodeca(d));
  f = (f!=undef?f:dodeca_faces());
  if (n == 0) polyhedron(v, f, convexity=10);
  else {
    r = d/2;
    s = len(v);
    vv = concat(v, [for (t=f) let(c=centroid3d([for (i=t) v[i]])) c*r/norm(c)]);
    ff = [for (i=incline(f)) for (j=incline(f[i])) [s+i, f[i][j], f[i][(j+1)%5]]];
    icosabouquet(d, n-1, vv, ff);
  }
}

// ====================================================================
// modeling tools
// ====================================================================

// peek value, or override value, for debugging functions
function peek(exp, val) = let(e=ifundef(val, exp)) echo(peek=e) e;

// echo value conditionally for debugging, e.g. val=tee(a*b+c, a>b)
function tee(exp, label, enable=true) = enable ? echo(label ? str(label, "=", strc(exp)) : exp) exp : exp;

// change color and opacity of children
module red(a=1) color("red", alpha=a) children();
module green(a=1) color("green", alpha=a) children();
module blue(a=1) color("blue", alpha=a) children();
module cyan(a=1) color("cyan", alpha=a) children();
module gold(a=1) color("gold", alpha=a) children();
module pink(a=1) color("pink", alpha=a) children();
module black(a=1) color("black", alpha=a) children();
module yellow(a=1) color("yellow", alpha=a) children();
module magenta(a=1) color("magenta", alpha=a) children();

// change opacity of children
module fade(alpha=0.5) color(alpha=alpha) children();

// list adjacent duplicates
module lsdup(array) {
  s = seams(array);
  if (len(s)==0) echo("No dup");
  else echo(dup=s, count=len(s), size=len(array));
}

// list bad points in array or mesh
module lsbad(array, layer=0) {
  for (i=[0:len(array)-1]) {
    e = array[i];
    if (rank(e)>1) lsbad(e, layer+1);
    else if (len(e)==2 && !valid2d(e)) echo(layer=layer, i=i, e);
    else if (len(e)==3 && !valid3d(e)) echo(layer=layer, i=i, e);
  }
}

// echo each element on a separate line
module echoln(array) {
  for (i=[0:len(array)-1]) echo(i, array[i]);
}

// echo string in color
module alert(msg, color="red") {
  echo(strc(msg, color));
}

// echo a colored string if debugging is enabled ($debug=true)
module debug(s1, s2, color="red") {
  if ($debug) {
    msg = s2 ? str(s1, s2) : s1; 
    echo($NOW ? str("[", $NOW, "] ", msg) : strc(msg, color=color));
  }
  children();
}
// switchable hiding of children
module hide(enable=true) {
  if (!enable) children();
}

// switchable highlight on children
module highlight(enable=true) {
  if (enable) #children(); else children();
}

// switchable ghosting of children
module ghost(enable=true) {
  if (enable) %children(); else children();
}

// switchable coloring of children
module paint(c, alpha=1, enable=true) {
  if (enable) color(c=c, alpha=alpha) children(); else children();
}

// select one of the children (0..$children-1) or everything if idx is undef
module child(idx) {
  if (idx==undef) children(); else if (idx>=0 && idx<$children) children(idx);
}

// a union of the enumerated children (e.g. enum=[0,3]) or everything if enum is undef
module select(enum) {
  if (enum==undef) children();
  else for (i=enlist(enum)) if (i>=0 && i<$children) children(i);
}

// random color for children
module rnd_color(seed) {
  s = rnd_seed(seed);
  for (i=[0:$children-1]) color(palette(i, seed=i+s/(i+1))) children(i);
}

// paint each child with a different palette color, s=starting color
module colorize(s=0) {
  for (i=[0:$children-1]) color(palette(s+i)) children(i);
}

// rounding a 2D polygon (not a profile)
module unsharp(amount) {
  if (amount==0) children(); else offset(amount) offset(-amount) children();
}

// vertical shafts at a list of locations
// zz=[min,max] along z axis, m=diameter, h=head countersink, t=tail countersink, g=gap
module shaft(locs=[[0,0]], zz=[-0.01,10], m=3, h, t, g=0, debug) {
  zz = is_list(zz) ? zz : [0,zz];
  for (p=locs) highlight(debug) hole(p, zz, m, h, t, -g, ext=0);
  children();
}

// make a hole shaped by profile in a list of locations, zz=[min,max] height range, e=clean cut allowance
module oust(profile, locs=[[0,0]], zz=[-10,10], radiate=1, e=0.01, enable=true, debug=false) {
  zz = zz[1]?zz:[0,zz?zz:10];
  if (enable && profile != undef) difference() {
    children();
    highlight(debug) radiate(radiate) for (p=locs) 
      translate([p[0],p[1],min(zz[0],zz[1])-e]) solid(profile, h=abs(zz[1]-zz[0])+e*2);
  }
}

// punch a set of vertical holes on the children with optional pipe around the hole
// locs=list of v2 points, zz=[min,max] along z axis, m=diameter, 
// h=head countersink, t=tail countersink, g=gap, wall=optional thickness of pipe around hole
module punch(locs=[[0,0]], zz=[-10,10], m=3, h, t, g, wall, enable=true, debug=false) {
  zz = zz[1]?zz:[0,zz?zz:10];
  difference() {
    union() {
      children();
      if (enable && m>0 && wall != undef) for (p=locs) 
        translate([p[0],p[1],min(zz[0],zz[1])]) cylinder(d=m+wall*2, h=abs(zz[1]-zz[0]), $fn=_fn(m/2));
    }
    if (enable && m>0) for (p=locs) highlight(debug) hole(p, zz, m, h, t, g);
  }
}

// more flexible than punch(), drill holes at a list of locations, default direction is along y-axis
// locs=list of v3 points, xx=[min,max] along x axis, etc. relative to each point, m=diameter
// h=head countersink, t=tail countersink, g=gap
module drill(locs=[[0,0,0]], xx, yy, zz, m=3, h, t, g=0, enable=true, debug=false) {
  difference() {
    children();
    if (enable && m>0) {
      yy = (xx==undef && yy==undef && zz==undef) ? [-50,50] : yy; // default
      rx = ifundef(xx, [0,0]);
      ry = ifundef(yy, [0,0]);
      rz = ifundef(zz, [0,0]);
      ph = [rx[0],ry[0],rz[0]];
      pt = [rx[1],ry[1],rz[1]];
      pp = ph-pt;
      for (p=locs) highlight(debug) translate(p+pt) orient(pp) hole([0,0], [0,norm(pp)], m, h, t, g);
    }
  }
}

// negative space for a vertical screw (see punch())
// head,tail=countersink [outer diameter, outer depth, inner depth] (e.g. [6,0.5,2] for M3)
// ext=extension on both ends to allow a clean cut
module hole(xy=[0,0], zz=[0,3], m=3, head, tail, gap, ext=0.1) {
  mm = m + ifundef(gap, 0.2); // compensate for printed hole shrinking
  translate([xy[0],xy[1],zz[0]-ext]) {
    xo = mm/2-ext-0.01;
    $fn = mof(_fn(m/2));
    cylinder(d=mm, h=abs(zz[1]-zz[0])+ext*2);
    if (head != undef) translate([0,0,zz[1]-zz[0]+ext*2]) rotate_extrude(angle=360, convexity=9)
      polygon([[xo,0],[head[0]/2+ext,0],[head[0]/2+ext,-head[1]-ext*2],[xo,-head[2]-ext*2]], convexity=9);
    if (tail != undef) rotate_extrude(angle=360, convexity=9)
      polygon([[xo,0],[tail[0]/2+ext,0],[tail[0]/2+ext,tail[1]+ext*2],[xo,tail[2]+ext*2]], convexity=9);
  }
}

// cut a hole in children for USB micro B socket, a=spin (0 means along y-axis)
module cut_usb_hole(p=[0,0,0], a=0, depth=3, debug=false) {
  difference() {
    children();
    highlight(debug) translate(p) rotate([0,0,a]) usb_micro_hole(depth);
  }
}

// side hole for a standard USB mini B (old) socket
module usb_hole(depth=3) {
  dd = depth + 0.2; // cut-out margin
  usb_mini_b = [[4,4],[4,2],[3,0],[-3,0],[-4,2],[-4,4]];
  translate([0,dd/2-0.1,0]) rotate([90,0,0]) linear_extrude(dd, convexity=9) polygon(usb_mini_b, convexity=9);
}

// side hole for a standard USB micro B socket
module usb_micro_hole(depth=3) {
  dd = depth + 0.2; // cut-out margin
  usb_micro_b = [[4,3.4],[4,1],[3,0],[-3,0],[-4,1],[-4,3.4]];
  translate([0,dd/2-0.1,0]) rotate([90,0,0]) linear_extrude(dd, convexity=9) polygon(usb_micro_b, convexity=9);
}

// PICOK 200 perfboard
module breadboard(w=80, d=50, h=1.5, r=0.5) {
  punch([[w/2-3,0],[-w/2+3,0]], [0,2]) plate(quad_path(w, d), h, r=r);
}

// vertical screw thread added to children
// m=screw_diameter, h=[start_height, end_height], b=[bottom_space, top_space], open=hole_size,
// taper=thread_tapering, flat=flat_top, v=thread_depth_scaler
module screw_thread(m=3, pitch=0.5, h=[0,10], b=[0,0], gap=0, open=0, taper=true, flat=true, v=1, debug=false) {
  p = pitch;
  h0 = is_list(h) ? h[0] : 0;
  h1 = is_list(h) ? h[1] : h;
  hh = h1-h0;
  b0 = is_list(b) ? b[0] : b; // bottom non-threaded space
  b1 = is_list(b) ? b[1] : 0; // top non-threaded space
  bb = hh-b0-b1;
  td = v*p*sqrt(3)/2; // depth of thread
  ppr = _fn(m/2); // points per revolution
  r = td/(p/2);
  margin = 0.02;
  xsec = [[-margin,0,0.75*td/r], [td*5/8,0,p/16], [td*5/8,0,-p/16], [-margin,0,-0.75*td/r]];
  mesh = [for (t=quanta(ppr*bb/p, end=1.01)) let(a=t*bb*360/p+(h0+b0)*360/p) shift3d(spin3d(xsec*(taper ? scale_guide(t) : 1), a), [(m/2-td*5/8-gap)*cos(a),(m/2-td*5/8-gap)*sin(a),t*bb])];
  highlight(debug) ascend(h0) intersection() {
    union() {
      if (hh>0 && bb>0 && td>0) ascend(b0) layered_block(mesh);
      if (open==0) cylinder(d=m-td*5/4-gap*2, h=hh, $fn=ppr);
      else if (open<m) pipe(d=m-td*5/4-gap*2, h=hh, m=open, flat=flat);
    }
    cylinder(d=m-gap*2+margin*2, h=hh+0.01, $fn=ppr);
  }
  children();
}

// vertical nut thread inside a ring or, if given, carved from children
// m=[screw_diameter, ring_diameter], h=[start_height, end_height], b=[bottom_space, top_space]
module nut_thread(m=[3,5], pitch=0.5, h=[0,10], b=[0,0], gap=0.4, v=1, debug=false) {
  d0 = is_list(m) ? m[0] : m;
  d1 = is_list(m) ? m[1] : d0+gap*2+0.02;
  h0 = is_list(h) ? h[0] : 0;
  h1 = is_list(h) ? h[1] : h;
  hh = h1-h0;
  b0 = max(0, is_list(b) ? b[0] : b); // bottom non-threaded space
  b1 = max(0, is_list(b) ? b[1] : 0); // top non-threaded space
  bx = [b0>0 ? b0 : b0-pitch, b1>0 ? b1 : b1-pitch]; // fully open when b<=0
  difference() { // carve from children or a cylinder
    if ($children) children(); else ascend(h0) cylinder(d=d1, h=hh);
    highlight(debug) screw_thread(m=d0, pitch=pitch, h=[h0-0.01,h1+0.01], b=bx, gap=-gap, open=0, v=v);
  }
}

// thread on a bottle: d=neck_diameter, w=thread_dimensions, b=vertical_bias, a=spin (for cap), n=count,
// cap=cap_mode, gap=spacing, c=thread_lead
// when n>1 and !cap, each thread bends down to provide a stop for the lug neck finish
module bottle_thread(d, pitch=4, h=[0,10], w=[1.2,0.8], b=0, a=0, n=1, cap=false, gap=0, c=2) {
  w0 = opt(w, 0);
  w1 = opt(w, 1);
  r = d/2 + (cap ? w0+gap+0.3 : -gap-0.2);
  h0 = is_list(h) ? h[0]-b : -b;
  h1 = is_list(h) ? h[1]-b : h-b;
  hh = h1-h0;
  zp = pitch==0; // zero pitch
  c = zp ? 0 : c;
  k = zp ? 1 : hh/pitch; // how many rounds
  g = zp ? ring_path(d) : [let(d=k*d*PI, a0=360*h0/pitch) for (t=quanta(ceil(d/$fs))) let(i=a0+t*k*360) [r*cos(i),r*sin(i),b+h0+hh*t-(cap||n==1?0:w1*2*max(0,1-d*t/6)^7)]];
  radiate(n) spin(a) translate([0,0,pitch*a/360+(n==1?0:w1*2.8)]) {
    fillet_sweep(round_path(w0+0.2, w1, cap?[90,270]:[-90,90]), g, c0=c, c1=c, twist=false, loop=zp);
    if (!cap && n>1) let(e=g[len(g)-1]) orient(force2d(e)) slide(x=r+0.1) flipy(90, e[2]-w1*2-0.3) cookie_extrude(ring_path(w1*2), w0*0.8); // bump
  }
  children();
}

// thread on a cap (unlike nut_thread, not a negative space), to pair with bottle_thread()
module cap_thread(d, pitch=4, h=[0,10], w=[1.2,0.8], b=0, a=0, n=1, gap=0.2, c=3) {
  w0 = opt(w, 0);
  w1 = opt(w, 1);
  bb = b-sqrt(3)*(w0-0.2)*w1/w0;
  bottle_thread(d=d, pitch=pitch, h=h, w=w, b=bb, a=a, n=n, cap=true, gap=gap, c=c) children();
}

// lug threads on a bottle: d=neck_diameter, w=thread_dimensions, b=baseline, a=spin, n=count, cap=cap_mode, gap=spacing
module bottle_lugs(d, pitch=1, w=[1.2,0.8], b=0, a=0, n=4, cap=false, gap=0) {
  w0 = opt(w, 0);
  w1 = opt(w, 1);
  bb = cap ? b-sqrt(3)*(w0-0.05)*w1/w0+0.1 : b;
  n = max(2, n); // at least 2
  h0 = cap ? pitch*w1*2.8/(PI*d) : 1e-3;
  h1 = pitch*(1/(n*2)-(cap?2.1:1.1)/(PI*d));
  bottle_thread(d=d, pitch=pitch, h=[h0,h1], w=w, b=bb, a=a+360*bb/pitch, n=n, cap=cap, gap=gap) children();
}

// lug threads on a cap, to pair with bottle_lugs()
module cap_lugs(d, pitch=1, w=[1.2,0.8], b=0, a=0, n=4, gap=0) {
  bottle_lugs(d=d, pitch=pitch, w=w, b=b, a=a, n=n, cap=true, gap=gap) children();
}

// cut nut holes at a list of locations
module nut_holes(locs=[[0,0,0]], m=3, pitch=0.5, h=5, gap=-0.4, e=[0,0,0]) {
  difference() {
    children();
    clone_at(locs) rotate(e) screw_thread(m=m, pitch=pitch, h=[-0.001,h+0.001-gap], gap=gap);
  }
}

// an insert for a screw, m=screw_diameter, h=height, b=solid_range, pitch=screw_pitch, gap=extra_gap, xz=fillet_size
module screw_prop(m=[3,5], h=[0,10], b=[0,0], pitch=0.5, gap=0.1, xz=[1,2]) {
  m0 = is_list(m) ? m[0] : m;
  m1 = is_list(m) ? m[1] : m+3;
  h0 = is_list(h) ? h[0] : 0;
  h1 = is_list(h) ? h[1] : h;
  b0 = is_list(b) ? b[0] : b;
  b1 = is_list(b) ? b[1] : 0;
  hh = h1 - h0;
  nut_thread(m, pitch=pitch, h=[h0+b0,h1-b1], b=[0,0], gap=gap)
    ascend(h0) fillet_pipe(m1, h=hh, xz=xz, m=0);
}

// an insert for heatset nuts, d=outer_diameter, h=height, m=screw_diameter, w=inner_diameter (3.5 for M2, 4.2 for M3)
module heatset_prop(d=8, h=3, m=3, w=4.2, fillet=2) {
  fillet_pipe(d=max(m+4, d), h=h, m=w-0.2, xz=[max(0,opt(fillet,0)),opt(fillet,1,3)]);
  if (h>3) pipe(d=w+2, h=h-3, m=m+0.2);
}

// an insert for M4 x 1 magnet with slightly narrow rim
module magnet_prop(h=1.4, d=4.3) {
  $fs=0.4;
  ascend(h<0?-0.01:-h+0.01) fillet_extrude(ring_path(d), h=abs(h), xz0=[h<0?-0.2:0,0.4], xz1=[h<0?0:-0.2,0.4]);
}

// clip within a volume defined by dm above xy-plane, then sliced at cx, cy and cz 
module clip(dm=[150,150,150], cx, cy, cz, origin, enable=true, margin=0.5, debug=false) {
  intersection() {
    children();
    if (enable) highlight(debug) translate([0,0,dm[2]/2]+(origin==undef?[0,0,0]:origin)) intersection() {
      c = [cx==undef?0:cx-dm[0]/2-margin,cy==undef?0:cy+dm[1]/2+margin,cz==undef?0:cz-dm[2]-margin];
      translate(c) cube(dm+[margin,margin,margin]*2, center=true);
      cube(dm+[margin,margin,margin]*2, center=true);
    }
  }
}

// remove anything above horizontal plane at z=h
module clip_ceiling(h=0, w=150, depth=150, enable=true, debug=false) {
  if (enable) intersection() {
    children();
    translate([-w/2,-w/2,h-depth]) highlight(debug) cube([w,w,depth]);
  }
  else children();
}

// remove anything below horizontal plane at z=h
module clip_floor(h=0, w=150, depth=150, enable=true, debug=false) {
  if (enable) intersection() {
    children();
    translate([-w/2,-w/2,h]) highlight(debug) cube([w,w,depth]);
  }
  else children();
}

// keep only a horizontal slice
module clip_slice(zz=[0,10], w=150, enable=true, debug=false) {
  if (enable) intersection() {
    children();
    translate([-w/2,-w/2,zz[0]]) highlight(debug) cube([w,w,zz[1]-zz[0]]);
  }
  else children();
}

// clip out anything closer than xz-plane
module clip_xz(cut=0, w=200, depth=150, enable=true, debug=false) {
  if (enable) intersection() {
    children();
    highlight(debug) translate([-w/2,-cut,-w/2]) cube([w,depth+cut,w]);
  }
  else children();
}

// clip out anything with x>0
module clip_yz(cut=0, w=200, depth=150, enable=true, debug=false) {
  if (enable) intersection() {
    children();
    highlight(debug) translate([-depth,-w/2,-w/2]) cube([depth+cut,w,w]);
  }
  else children();
}

// chop horizontally into 2 objects
module chop(h=0, apart=100, w=150, depth=150, debug=false, select) child(select) {
  translate([select!=undef?0:-apart/2,0,apart==0?0.2:-h]) clip_floor(h=h+0.01, w=w, depth=depth, debug=debug) children();
  translate([select!=undef?0:apart/2,0,0]) clip_ceiling(h=h-0.01, w=w, depth=depth, debug=debug) children();
}

// scale individual dimensions of children (size dm) by absolute amount defined by sm
module scale_by(dm, sm, origin) {
  c = ifundef(origin, [0,0,dm[2]/2]);
  sm = rank(sm)==0 ? [sm,sm,sm] : as3d(sm);
  sx = dm[0]==0?1:(dm[0]-sm[0]*2)/dm[0];
  sy = dm[1]==0?1:(dm[1]-sm[1]*2)/dm[1];
  sz = dm[2]==0?1:(dm[2]-sm[2]*2)/dm[2];
  translate(c) scale([sx,sy,sz]) translate(-c) children();
}

// make object centered at origin to become hollow by subtracting a scaled down copy (3D offset where are you?)
module hollow(dm, wall=2, origin) {
  if (dm != undef) {
    difference() {
      children();
      if (wall!=0) scale_by(dm, wall, origin) children();
    }
  }
}

// make a case given a child of convex solid
module make_case(dm, wall=2, cut=0, apart=100, flip=true, origin, debug=false) {
  clip_xz(w=max(snip(dm*2)), depth=dm[1], enable=debug) scatter(debug ? 0 : apart) {
    cut_lid(dm, wall=wall, cut=cut, flip=flip, apart=apart, origin=origin, debug=debug) children();
    cut_bin(dm, wall=wall, cut=cut, flip=flip, apart=apart, origin=origin, debug=debug) children();
  }
}

// see make_case()
module cut_lid(dm, wall, cut=0, apart=100, flip=true, origin, debug=false) {
  ascend(debug || !flip ? 0 : dm[2]) flipy(debug || !flip ? 0 : 180) ascend(debug ? 0.2 : flip ? 0 : -cut+3) {
    w = max(dm[0],dm[1])+10;
    clip_floor(cut, w=w, depth=dm[2]+10) hollow(dm, wall+0.2, origin) children();
    clip_slice([cut,dm[2]], w=w) hollow(dm-[wall,wall,0.01], [wall,wall,0.01], origin)
      scale_by(dm, [wall+0.1,wall+0.1,0]) children();
    ascend(cut-2.99) deepen(3) projection(true) ascend(-cut)
      hollow(dm-[wall-0.1,wall-0.1,0.01], [wall-0.1,wall-0.1,0.01], origin)
      scale_by(dm, [wall+0.2,wall+0.2,0]) children();
  }
}

// see make_case()
module cut_bin(dm, wall, cut=0, apart=100, flip=true, origin, debug=false) {
  w = max(dm[0],dm[1])+10;
  clip_ceiling(cut, w=w, depth=dm[2]+10) hollow(dm, wall, origin) children();
}

// make a case with lid from any profile (see also basic_case, cover_case in object.scad)
// h=case height, b=bin height, t=thickness, j=joiner height, g=joiner gap
module case_extrude(profile, h=30, b=24, t=2.8, j=4, g=0.1, bin=true, lid=true, debug=false) {
  sp = span(minmax(slice(profile, 0))) + t*2 + 20;
  clip_xz(enable=debug) scatter(sp, enable=!debug) {
    if (bin) {
      basin(profile, b-g, -t);
      children();
    }
    if (lid) flipy(h=h, enable=!debug) ascend(h) {
      ascend(-h+b+0.01-j) shell(profile, h-b-t+j, -1.2, inflate=-t-g); // joiner
      //ascend(-h+b+0.01) shell(profile, h-b-t, -t-1.2); // outer
      ascend(-h+b+0.02) deepen(h-b-t) difference() { // outer
        polygon(profile);
        offset(-t-1.19) polygon(profile);
      }
      ascend(-t) fillet_extrude(profile, t, [0,0], [-1,2]); // top
    }
  }
}

// extrude a profile with a rounded top, h=height, inset=amount to shrink, b=height of vertical base
module cookie(profile, h=5, inset, b=0) {
  s = (inset!=undef?inset:abs(h));
  b = min(abs(h), abs(b));
  r = max(0, abs(h)-b) * sign(h);
  m = s==0 ? [profile] : [for (c=round_path(r, s, a=[90,0])) force3d(reverse(offset2d(profile, c[1]-s), h<0), c[0])];
  layered_block(b==0 ? m : prepend(shift_layers(m, [0,0,h<0?-b:b]), m[0]));
}

// extrude a profile with a rounded top, b=height of vertical wall, origin=peak location, f=obesity
module cookie_extrude(profile, h, b=0, origin=[0,0], f=0) {
  layered_block(morph_cookie(profile, h, b, origin, f));
}

// create a dome from a profile, t=thickness, vault=number of vaults for support, v=vault diameter
module cookie_dome(profile, h, b=0, t=1.6, origin=[0,0], vault=0, v=1) {
  m = morph_cookie(profile, h=h-t, b=b, origin=origin);
  difference() {
    cookie_extrude(offset2d(profile, t), h=h, b=b, origin=origin);
    ascend(-0.01) layered_block(m);
  }
  if (vault>0) for (s=every(isomesh(m), len(m[0])/vault)) trace(snip(s, 1, 2), d=v, r=0.2);
}

// extrude from a SVG or DXF file
// file=import path, xsize,ysize=dimensions, h=height, xoff,yoff=offsets, thicken=size adjustment
module file_extrude(file, h=1, xsize=undef, ysize=undef, xoff, yoff, inflate=0, deflate=0) {
  xo = xoff == undef ? 0 : xoff;
  yo = yoff == undef ? 0 : yoff;
  translate([xo,yo]) linear_extrude(h, convexity=9)
    offset(-deflate) offset(inflate) resize([xsize,ysize], auto=true) import(file, center=true);
}

// extrude a stamp from a SVG or DXF file
module stamp_extrude(file, xsize=50, step=0.2, layers=5, slope=0.5, inflate=0) {
  if (slope == 0)
    ascend(step*layers) flipy() file_extrude(file, step*layers, xsize, inflate=inflate);
  else
    for (i=[0:layers]) ascend(i*step-0.001) flipy() file_extrude(file, step+0.002, xsize, inflate=inflate+(1-i/layers)*slope);
}

// cut a SVG or DXF outline of width w and depth d centered at point p
module file_imprint(file, p=[0,0,0], w=undef, d=0.4, inflate=0, debug=false) {
  difference() {
    children();
    highlight(debug) translate(p-[0,0,d-0.001]) file_extrude(file, h=d, xsize=w, inflate=inflate);
  }
}

// add a SVG or DXF outline of width w and depth d centered at point p
module file_emboss(file, p, w, d=0.4, inflate=0, color) {
  children();
  color(color) translate(p-[0,0,0.001]) file_extrude(file, h=d, xsize=w, inflate=inflate);
}

// generate a mesh block from a set of mxn points representing a rectangular surface above x-y plane
// points=[L1, L2, L3, ... Lm] where each L is a list of 3D points [p1, p2, p3, ... pn] of length n
module mesh_block(points, h=0, zscale=1) {
  xsteps = len(points[0])-1;
  xlow = points[0][0][0];
  xhigh = points[0][xsteps][0];
  ysteps = len(points)-1;
  ylow = points[0][0][1];
  yhigh = points[ysteps][0][1];
  idx = (xsteps+1) * (ysteps+1);

  mesh = [for (y=[0:ysteps]) shift3d(scale3d(points[y],[1,1,zscale]),[0,0,h])];
  corners = [[xlow,ylow,0],[xhigh,ylow,0],[xhigh,yhigh,0],[xlow,yhigh,0]];

  top = [for (j=[0:ysteps-1], i=[0:xsteps-1], f=[1,0]) f ?
    [j*(xsteps+1)+i, (j+1)*(xsteps+1)+i, j*(xsteps+1)+i+1] :
    [j*(xsteps+1)+i+1, (j+1)*(xsteps+1)+i, (j+1)*(xsteps+1)+i+1]
  ];

  base = [
    concat([for (i=[0:xsteps]) i], [idx+1, idx, 0]),
    concat([for (i=[0:xsteps]) idx-i-1], [idx+3, idx+2, idx-1]),
    concat([for (i=[1:ysteps+1]) idx-i*(xsteps+1)], [idx, idx+3, idx-xsteps-1]),
    concat([for (i=[0:ysteps]) (i+1)*(xsteps+1)-1], [idx+2, idx+1, xsteps]),
    [idx, idx+1, idx+2, idx+3]
  ];
  polyhedron(flatten(concat(mesh, [corners])), concat(top, base), convexity=9);
}

// generate a mesh block from a set of points representing n layers of contours (no need to be coplanar)
// layers=[L1, L2, ... Lm] where each L is a list of 3D points [p1, p2, p3, ... pn] of same length n
module layered_block(layers, loop=false, invert=false) {
  k = len(layers);
  n = len(layers[0]);
  if (k>0 && n>0) {
    v = [for (i=indices(layers, invert)) each layers[i]];
    g = len(v);
    f = [for (j=[0:k-(loop?1:2)], i=[0:n-1]) let(a=n*j, b=n*((j+1)%k), c=(i+1)%n) each [[a+i,b+c,a+c],[a+i,b+i,b+c]]];
    polyhedron(v, loop ? f : concat(f, [[for (i=[0:n-1])i], [for (i=[1:n])g-i]]), convexity=10);
  }
}

// create a layered shell by thickening each layer outward (or inward if negative), requires coplanar layers
// skip=how many layers to skip at the bottom, to create a floor
module layered_shell(layers, t=1.6, skip=0, flat=true) {
  if (t!=0) {
    e = len(layers)-1;
    m1 = [for (i=[e:-1:0]) let(l=layers[i], c=offset2d(l, t)) [for (j=[0:len(l)-1]) [c[j][0],c[j][1],l[j][2]]]];
    m2 = flat ? [] : [let(p=layers[e], h=p[0][2], tt=abs(t)) for (i=snip(arc_path(tt, [0,180]), 1, 1)) force3d(offset2d(p, t/2-i[0]), h+i[1])];
    layered_block(snip(reverse(concat(layers, m2, m1), enable=t>0), skip), loop=skip==0);
  }
}

// create a dome-shape layered shell using an inner wall formed by offseting the layers inward
// skip=how many layers to skip at the top, to control its thickness
module layered_dome(layers, inner=0.8, skip=3) {
  p = concat([for (l=reverse(snip(layers, skip))) expand2d(l, -inner)], layers);
  layered_block(p);
}

// knit a mesh on the points surface, t=thread_thickness, d=dilution, s=twist (e.g. [-1,0,1])
module layered_mesh(layers, t=1.6, d=1, s=0) {
  for (i=enlist(s), p=isomesh(layers, d, i)) trace(p, t);
}

// plot layers
module plot_layers(layers, intervals, d=0.2, loop=false, color, dup=false) {
  if (intervals==undef)
    for (l=layers) plot(l, d=d, loop=loop, color=l==layers[0]?"blue":color, dup=dup);
  else {
    c = [let(z=accum(intervals)) for (i=indices(layers)) force3d(layers[i], z[i])];
    plot_layers(c, d=d, loop=true, color=color?color:"red");
  }
}

// plot slopes
module plot_slopes(layers, d=0.2, color, dup=false) {
  for (k=isomesh(layers)) plot(k, d=d, loop=false, color=color, dup=dup);
}

// sweep a profile along 3D path, scaler=[[t,f],...] where t=proportion in [0,1], f=scaling factor, s=twist
module sweep(profile, path, scaler, s=0, loop=false) {
  if (profile==undef || path==undef)
    debug("Error: sweep() - profile and path parameters required");
  else if (len(path)>1) {
    p = force3d(path);
    if (scaler==undef && len(path[0])==2) // 2D path
      layered_block(sweep_wall(profile, p, loop=loop, s=s), loop=loop);
    else if (scaler==undef)
      layered_block(sweep_profile(profile, p, loop=loop, s=s), loop=loop);
    else {
      layers = [for (i=quanta(len(p)-1, end=1)) profile*lookup(i, scaler)];
      layered_block(sweep_layers(layers, p, loop=loop, s=s), loop=loop);
    }
  }
}

// sweep a profile along 3D path with pointy ends (like a calligraphy stroke), f=thickness control
module sweep_stroke(profile, path, f=1, flat=false) {
  layers = [for (t=quanta(len(path)-1, end=1)) let(s=sin(t*180)^f) flat ? scale2d(profile, [s,1]) : profile*s];
  layered_block(sweep_layers(layers, force3d(path), loop=false, s=0), loop=false);
}

// ====================================================================
// placement modules
// ====================================================================

// raise up children
module ascend(z=0.001, enable=true) if (enable) translate([0,0,z]) children(); else children();

// similar to translate but defaults to zero for undefined axes
module slide(x=0, y=0, z=0) translate([x,y,z]) children();

// rotate children about an axis, then ascend by h
module flipx(a=180, h=0, enable=true) if (enable) ascend(h) rotate([a,0,0]) children(); else children();
module flipy(a=180, h=0, enable=true) if (enable) ascend(h) rotate([0,a,0]) children(); else children();
module flipz(a=180, h=0, enable=true) if (enable) ascend(h) rotate([0,0,a]) children(); else children();

// rotate angle a about z-axis
module spin(a, origin=[0,0]) translate(origin) rotate([0,0,a]) translate(-origin) children();

// orient children using the transformation [0,0,1] -> v (or [1,0] -> v if v is 2D) 
module orient(v, from=[0,0,1], basis) {
  if (len(v)<3) spin(atan2(v[1], v[0])) children();
  else multmatrix(mm_rotate(v, from, basis)) children();
}

// randomly rotate children
module random_rotate(seed) {
  s = rnd_seed(seed);
  for (i=[0:$children-1]) rotate(rnd(0,360,3,i+s/i)) children(i);
}

// randomly spin children
module random_spin(seed) {
  s = rnd_seed(seed);
  for (i=[0:$children-1]) rotate([0,0,rnd(0,360,1,i+s/i)]) children(i);
}

// scatter children evenly along a guiding path, e.g. [[-10,0,0],[10,0,0]]
module spread(path) {
  path = (path!=undef?path:ring_path(100));
  for (i=[0:$children-1]) translate(path_lookup(path, i/$children)) children(i);
}

// scatter children in a 1, 2, or 3 dimensional grid centered at origin
module scatter(x=0, y=0, z=0, copy=0, enable=true) {
  d = len(eliminate([x,y,z], 0));
  if (!enable || $children==0 || d==0 || copy==1) children(); // do nothing
  else {
    c = copy>0 ? copy : $children;
    m = floor([0,c,sqrt(c),pow(c, 1/3)][d]);
    nx = x==0 ? 1 : (d==2 && m*m<c) || (d==3 && m*m*m<c) ? m+1 : m;
    ny = y==0 ? 1 : (d==2 && m*m<c && nx*m<c) || (d==3 && m*m*m<c && nx*m*m<c) ? m+1 : m;
    nz = z==0 ? 1 : ceil(c/nx/ny);
    cx = (nx-1)*x/2;
    cy = (ny-1)*y/2;
    cz = (nz-1)*z/2;
    for (k=[0:nz-1],j=[0:ny-1],i=[0:nx-1]) let(n=i+j*nx+k*nx*ny)
      if (copy>0 && n<c) translate([i*x-cx,j*y-cy,k*z-cz]) children();
      else if (n<c) translate([i*x-cx,j*y-cy,k*z-cz]) children(n);
  }
}

// scatter children along a Ulam spiral on xy-plane
module ulam_scatter(xy) {
  sx = is_list(xy) ? xy[0] : xy;
  sy = is_list(xy) ? xy[1] : xy;
  for (i=[0:$children-1]) let(p=ulam(i)) translate([p[0]*sx,p[1]*sy]) children(i);
}

// scatter children along a smooth path evenly (n==undef && $children>1), repeatedly (n>$children),
// or replicate a single object on every point of path (n==undef && $children==1)
// only up to 20 children can be provided (arbitary choice to overcome the implict union in OpenSCAD)
module path_scatter(path, n, loop=false, k, i=0, c=0, nc, a, m, t) {
  k = ifundef(k, len(path));
  if (i<k) {
    a = (a!=undef?a:mm_pitch(-sweep_twist(path, loop, k)));
    n = ifundef(n, $children==1 ? k : $children);
    nc = ifundef(nc, $children);
    tt = sweep_tangent(path, loop, k, i);
    mm = i==0 ? mm_rotate(tt, [0,-1,0]) : mm_rotate(tt, t)*m;
    hit = i==round(k*c/n);
    if (hit) translate(path[i]) multmatrix(mm) children(c%nc);
    if ($children==1) path_scatter(path, n, loop, k, i+1, hit ? c+1 : c, nc, a, mm*a, tt) children();
    else path_scatter(path, n, loop, k, i+1, hit ? c+1 : c, nc, a, mm*a, tt) {
      if ($children>0) children(0);
      if ($children>1) children(1);
      if ($children>2) children(2);
      if ($children>3) children(3);
      if ($children>4) children(4);
      if ($children>5) children(5);
      if ($children>6) children(6);
      if ($children>7) children(7);
      if ($children>8) children(8);
      if ($children>9) children(9);
      if ($children>10) children(10);
      if ($children>11) children(11);
      if ($children>12) children(12);
      if ($children>13) children(13);
      if ($children>14) children(14);
      if ($children>15) children(15);
      if ($children>16) children(16);
      if ($children>17) children(17);
      if ($children>18) children(18);
      if ($children>19) children(19);
    }
  }
}

// ====================================================================
// replication modules
// ====================================================================

// clone a number of copies on xy-plane
module clone(n, x=0, y=0, z=0) {
  scatter(x=x, y=y, z=(x==0 && y==0 && z==0 ? 10 : z), copy=n) children();
//translate([-((n-1)*spacing)/2,0,0]) for (i=[0:n-1]) translate([i*spacing,0,0]) children();
}

// clone at each of the 3D location, optional rot = rotation angles of each copy
module clone_at(points, rot) {
  if (rot == undef) {
    for (p=points) translate(p) children();
  } else {
    for (i=[0:len(points)-1]) translate(points[i]) rotate(rot[i%len(rot)]) children();
  }
}

// clip second child from first child at each of the 3D locations
module clip_at(points) {
  difference() {
    children(0);
    if ($children>1) for (p=points) translate(p) children(1);
  }
}

// replicate children in a radiating pattern on xy-plane
// r=radius, n=number of copies, hide=omit copies at the end
// $index is passed to each child for their information
module radiate(n, r=0, hide=0, spin=0) {
  if (n>0) {
    p = is_list(r) ? len(r) == 3 ? r : undef : [r,0,0];
    for (i=[0:n-hide-1]) rotate(360*i/n+spin) translate(p) { $index=i; children(); }
  }
}

// replicate children as mirror images across x and y planes (total of 4 copies)
module reflect() {
  children();
  mirror([1,0,0]) children();
  mirror([0,1,0]) children();
  mirror([1,0,0]) mirror([0,1,0]) children();
}

// generate kaleidoscope reflections of children (total of 6 copies)
module kaleidoscope() {
  children(); // original
  rotate(120) children();
  rotate(240) children();
  rotate(120) mirror([0,1,0]) children();
  rotate(240) mirror([0,1,0]) children();
  mirror([0,1,0]) children();
}

// displace, then replicate children as a mirror image across origin where m=axes, d=displacement
module mirror_clone(m=[0,1,0], d=[0,0,0]) {
  translate(d) children();
  mirror(m) translate(d) children();
}
module x_clone(d=[0,0,0]) { mirror_clone(m=[1,0,0], d=d) children(); }
module y_clone(d=[0,0,0]) { mirror_clone(m=[0,1,0], d=d) children(); }
module z_clone(d=[0,0,0]) { mirror_clone(m=[0,0,1], d=d) children(); }

// replicate children by radiate each copy at the points given, from=base axis
module orient_clone(points, from=[0,0,1]) {
  for (p=points) orient(p, from) children();
}

// clone a number of copies by applying a displacement d, an orientation to v, and a scaling s in each iteration
module mm_clone(n, d=[10,0,0], v=[0,0,0], s=[1,1,1], mm) {
  if (n>0) {
    mm = (mm!=undef?mm:mm_ident());
    multmatrix(mm) children();
    mm_clone(n-1, d, v, s, mm_scale(s) * mm_rotate(v) * mm_translate(d) * mm) children();
  }
}

// clone a number of copies along a Ulam spiral on xy-plane
module ulam_clone(n, xy=50) {
  sx = is_list(xy) ? xy[0] : xy;
  sy = is_list(xy) ? xy[1] : xy;
  for (i=[0:n-1]) let(p=ulam(i)) translate([p[0]*sx,p[1]*sy]) children();
}

// clone and scatter m by n copies on xy-plane
module grid_clone(m, n, xy=50) {
  sx = is_list(xy) ? xy[0] : xy;
  sy = is_list(xy) ? xy[1] : xy;
  s = [-sx*(m-1)/2, -sy*(n-1)/2];
  for (i=[0:m-1], j=[0:n-1]) translate([i*sx,j*sy]+s) children();
}

// clone and scatter n copies evenly over a sphere
module orbit_clone(n) {
  o = orbit(n);
  for (i=[0:n-1]) rotate(180,[0,0,1]+o[i]) { $index = i; children(); }// orbit() should never include [0,0,-1]
}

// clone n copies along a spiral path defined by d, a and f, start and end are in range [0,1]
module spiral_clone(n, d, a=[0,360], f=3, start=0, end=1) {
  let(a0=ifundef(a[0],0), a1=opt(a,1), aa=a1-a0)
    for (t=quanta(n, start=start, end=end)) let(b=a0+t*aa, e=pow(t,f)*d/2)
      translate([cos(b),sin(b)]*e) spin(t*aa) children();
}

// clone n children evenly along a guiding path, recycle if necessary, r=range for normal vector sampling
// copies are aligned with path points, so a fine path increases accuracy
// if there's only one child, then clone at each point of the (coarse) path
module sweep_clone(path, n, loop=true, r=1, debug=false) {
  if ($children>0) {
    k = len(path);
    c = centroid3d(path);
    n = ifundef(n, $children==1 ? k : $children);
    for (i=[0:n-1]) {
      j = round(k*i/n);
      if (debug) translate(path[j]) sphere(0.5);
      else {
        ax = unit(tangent_at(path, j, r, loop));
        ay = unit(-normal_at(path, j, r, loop, c));
        az = unit(cross(ax, ay));
        translate(path[j]) multmatrix(mm_reframe(ax, ay, az)) children(i%$children);
      }
    }
  }
}

// ====================================================================
// fillet modules
// ====================================================================

// a horizontal fillet bar of length d parallel to y-axis
module fillet_bar(d, xz=[2,2], convex=false) {
  translate([0,d/2]) rotate([90]) linear_extrude(d, convexity=9) polygon(fillet_path(xz[0],xz[1],convex), convexity=9);
}

// a fillet ring of diameter d, dimensions xz, and angle a
module fillet_ring(d, xz=[2,2], a=360, convex=false, core=false, vault=0, v=1) {
  loop = (a>=360);
  p = ring_path(abs(d), a=[0,a]);
  f = fillet_path(xz[0], xz[1], convex);
  radiate(vault) trace(swap_yz(shift2d(snip(f, 1, 2), [d/2,0])), d=v, r=0.2);
  sweep(reverse(f, sign(xz[0])!=sign(xz[1])), p, loop=loop);
  if (core) {
    p = ring_path(abs(d)+0.02, a=[0,a]);
    solid(loop ? p : append(p, [0,0]), xz[1]);
  }
}

// a rectangular fillet of size x by y and radius r
module fillet_rect(x, y, xz=[2,2], r=2, convex=false, core=false) {
  p = pad_path(x, y, r);
  f = fillet_path(xz[0], xz[1], convex);
  sweep(reverse(f, sign(xz[0])!=sign(xz[1])), p, loop=true);
  if (core) solid(p, xz[1]);
}

// a vertical pipe of diameter d, height h, thickness t, and bottom (or top, if invert is true) fillet of dimensions xz
module fillet_pipe(d, h=2, t=1, xz=[2,2], convex=false, invert=false, m) {
  r = d/2;
  t = m!=undef ? (d-m)/2 : t>0 ? min(r,t) : max(0,r+t);
  c = round_path(r);
  f = shift2d(subarray(fillet_path(xz[0], xz[1], convex), 1), [r,0]);
  mesh = concat([for (p=f) if (p[1]<=h) force3d(c*p[0]/r, p[1])], [force3d(c, h)],
      r-t<0.01 ? [] : concat(
        [force3d(c*(r-t)/r, h)],
        [force3d(c*(r-t)/r, min(h, xz[1]))],
        [force3d(c*(r-t)/r, max(0, -xz[0]))]
        ));
  layered_block(invert ? invert3d(mesh, h) : mesh, loop=(r-t>=0.01));
}

// a solid cube of dimensions dm and fillet radius at most r
module fillet_cube(dm, r=5, center=false) {
  sx = opt(dm, 0);
  sy = opt(dm, 1);
  sz = opt(dm, 2);
  rr = min(r, sx/2, sy/2, sz/2); // reduce radius accordingly
  if (rr > 0) {
    dv = ceil(rr/$fs);
    m0 = [for (j=quanta(dv, end=1)) let(s=rr*sin(j*90)) [for (i=[0:3])
        let(c=[(sx-rr*2)*(abs(i-1.5)-1),(sy-rr*2)*(i%2+1-i)/2, rr*(1-cos(j*90))], q=i*90)
        each [for (a=quanta(dv, end=1, max=90)) s*[cos(q+a),sin(q+a),0]+c]]];
    m1 = [for (s=reverse(m0)) [for (p=s) [p[0],p[1],sz-p[2]]]];
    ascend(center?-sz/2:0) layered_block(concat(m0, m1));
  }
}

// extrude a counterclockwise profile with configurable fillets at bottom (xz0) and top (xz1)
// the default will round both top and bottom (set x in xz to zero for no fillet)
module fillet_extrude(profile, h=1.6, xz0=[-1,1], xz1=[-1,1], r0, r1, r, convex=false, core=true) {
  if (h>0) {
    a0 = r0==undef ? r==undef ? abs(xz0[1]) : abs(r) : abs(r0);
    a1 = r1==undef ? r==undef ? abs(xz1[1]) : abs(r) : abs(r1);
    x0 = r0==undef ? r==undef ? xz0[0] : -r : -r0;
    x1 = r1==undef ? r==undef ? xz1[0] : -r : -r1;
    z0 = a0==0 || x0==0 ? 0 : min(h*a0/(a0+a1), a0);
    z1 = a1==0 || x1==0 ? 0 : min(h*a1/(a0+a1), a1);
    s0 = x0*z0==0; // straight
    s1 = x1*z1==0; // straight
    xx = min(0, s0?0:x0, s1?0:x1);
    c0 = [for(i=s0?[[0,0]]:fillet_path(x0, z0, convex, !core && x0>0)) force3d(offset2d(profile, i[0]), i[1])];
    c1 = [for(i=s1?[[0,0]]:fillet_path(x1, -z1, convex, !core && x1>0)) force3d(offset2d(profile, i[0]), i[1]+h)];
    cc = core ? [] : offset2d(profile, xx);
    m = concat([if (!core && (x1>=xx)) force3d(cc, 0)], c0, reverse(c1), [if (!core && (x0>=xx)) force3d(cc, h)]);
    layered_block(m, !core);
  }
}

// sweep profile along path with end caps defined by c0,c1=[r,e] where r=fillet radius, e=cap length (optional)
module fillet_sweep(profile, path, c0, c1, s=0, twist=true, loop=false, offset=false) {
  k = len(path);
  if (k>1) {
    r0 = opt(c0, 0); // start cap radius
    r1 = opt(c1, 0); // end cap radius
    e0 = abs(opt(c0, 1, r0)); // start cap length
    e1 = abs(opt(c1, 1, r1)); // end cap length
    a0 = [if (e0>0) let(fa=_fa(e0, r0), f=max(0,e0-r0)/e0) for (t=[0:fa:90-fa]) [cos(t)*e0,offset?(sin(t)-1)*r0:f+(1-f)*sin(t)]]; // start arc
    a1 = [if (e1>0) let(fa=_fa(e1, r1), f=max(0,e1-r1)/e1) for (t=[fa:fa:90]) [sin(t)*e1,offset?(cos(t)-1)*r1:f+(1-f)*cos(t)]]; // end arc
    c = concat([for (i=a0) offset?offset2d(profile, i[1]):profile*i[1]], [for (i=path) profile], [for (i=a1) offset?offset2d(profile, i[1]):profile*i[1]]);
    p = concat([let(b=path[0], u=unit(b-path[1])) for (i=a0) b+u*i[0]], path, [let(b=path[k-1], u=unit(b-path[k-2])) for (i=a1) b+u*i[0]]);
    layered_block(sweep_layers(c, force3d(p), s=s, twist=twist&&(s!=0||p[0][2]!=undef)), loop=loop);
  }
}

// create a lid from a profile with rounded top, where h=inner height, e=rim height, g=gap, top=closed or not
module fillet_lid(profile, h=5, t=1.6, e=3, g=0.1, top=true, flip=false, tidy) {
  tt = abs(t);
  td = t>0 && convex2d(profile) ? 0 : tidy;
  te = max(tt/2-g, 1);
  e = max(e, tt/2);
  p = scale2d(reverse(profile, loop=true), [-1,1]); // flip over
  pp = offset2d(p, -g-(t<0?tt:0), tidy=td);
  m1 = [for (i=arc_path(tt*2, [-90,0])) force3d(offset2d(p, i[0]-(t<0?tt:0), tidy=td), tt+i[1])];
  m2 = [force3d(last(m1), h+tt), force3d(pp, h+tt), force3d(pp, h+tt+e)];
  m3 = [for (i=arc_path(te*2, [0,90])) force3d(offset2d(pp, i[0]-te, tidy=td), h+tt+e+i[1])];
  flipy(enable=flip) {
    layered_block(concat(m1, m2, m3, [force3d(last(m3), top?tt:0)]), loop=!top);
    ascend(tt-0.01) children(); // allow union
  }
}

// create a bin from a profile with rounded bottom, where h=inner height, t=wall thickness (may be negative)
// children are added relative to the bottom inside the bin
module fillet_bin(profile, h=20, t=1.6, bottom=true, flat=true, tidy) {
  if (t!=0 && h!=0) {
    tt = abs(t);
    td = t>0 && convex2d(profile) ? 0 : tidy;
    m1 = [for (i=arc_path(tt*2, [-90,0])) force3d(offset2d(profile, i[0]-(t<0?tt:0), tidy=td), tt+i[1])];
    m2 = flat ? [force3d(last(m1), h+tt), force3d(m1[0], h+tt)] :
      [for (i=arc_path(tt, [0,180])) force3d(offset2d(profile, i[0]+t/2, tidy=td), h+tt/2+i[1])];
    layered_block(concat(m1, m2, [force3d(m1[0], tt)]), loop=!bottom);
    ascend(bottom?tt-0.01:0) children(); // allow union
  }
}

// create a tray of height h from a profile with rounded bottom r, negative thickness t means inner wall
// unlike fillet_bin() it supports deep rounding (limited only by h)
module fillet_tray(profile, h=20, t=1.6, r, bottom=true, flat=true) {
  if (t!=0 && h!=0) {
    tt = abs(t);
    rr = max(tt, min(abs(ifundef(r, t)), abs(h)-(flat?0:tt/2)));
    hh = flat ? abs(h) : abs(h)-tt/2;
    g = concat2d([r==0 ? [tt,0] : ccw_path([0,-rr], [rr,0], po=[0,0]),
      if (rr<hh-0.01) [0,hh-(r==0?0:rr)],
      flat ? [-tt,0] : ccw_path([tt,0], [0,0], po=[tt/2,0], $fs=$fs/2),
      if (rr<hh-0.01) [0,-hh+rr], cw_path([rr-tt,0], [0,tt-rr], po=[0,0])], [t<0?-rr:tt-rr,0]);
    layered_block([for (i=g) force3d(offset2d(profile, i[0]), i[1])], loop=!bottom);
    ascend(bottom?tt-0.01:0) children(); // allow union
  }
}

// create a dome of height h from a profile where t=thickness (may be negative, zero means solid), r=fillet radius
// and optionally a number of supporting vaults (v=diameter, s=shift positions) for 3D printing
module fillet_dome(profile, h=20, t=1.6, r=5, vault=0, v=1, s=0, tidy) {
  if (h>0) {
    tt = abs(t);
    r = min(r, h);
    v = min(v, tt);
    td = t>0 && tidy==undef && convex2d(profile) ? 0 : tidy;
    m0 = t==0 ? [] : r==0 ? [force3d(profile, h-tt)] : [for (i=arc_path((r-tt)*2, [90,0])) force3d(offset2d(profile, i[0]-r+tt-(t<0?tt:0), tidy=td), h-r+i[1])];
    m1 = r==0 ? [force3d(offset2d(profile, tt, tidy=td), h)] : [for (i=arc_path(r*2, [0,90])) force3d(offset2d(profile, i[0]-r+tt-(t<0?tt:0), tidy=td), h-r+i[1])];
    m = last(m0);
    layered_block(concat(m0, [if (t!=0) force3d(m, 0)], [force3d(m1[0], 0)], m1), loop=false);
    if (t!=0 && vault>0) {
      if (r-tt>=3) {
        k = len(m);
        q = isomesh(m0);
        for (i=[0:k-1]) let(a=abs((angle_at(m, i)+360)%360-180), j=(i+k+s)%k) if (a>15||ceil(floor(j*vault/k)*k/vault)==j) trace(snip(q[i], 1), d=v, r=0.3);
      }
      intersection() {
        layered_block(concat(m0, [force3d(m, 0)]));
        mx = minmax(slice(m, 0));
        my = minmax(slice(m, 1));
        for (y=[my[0]:5:my[1]]) trace([[mx[0],y,h-tt],[mx[1],y,h-tt]], d=max(1.2, v));
      }
    }
  }
}

// a box of inner dimensions dm, with rounded bottom and corners, negative t means dm is outer dimensions
module fillet_box(dm, t=1.6, r=5, bottom=true) {
  p = pad_path(dm[0], dm[1], r=min(min(redim(dm, 2))/2-0.01, r));
  fillet_bin(p, h=dm[2], t=t, bottom=bottom, tidy=0);
  ascend(t-0.01) children();
}

// a rectangular case with lid ready for 3D printing, see fillet_bin() and fillet_lid() for parameters,
// except here dm=[inner width, inner depth, inner height of bin, inner height of lid], debug=assembled view
// child 0 is added and child 1 is subtracted relative to inner bottom of bin (meant for lid=false or bin=false)
module fillet_case(dm, t=1.6, r=5, e=3, g=0.1, lid=true, bin=true, top=true, bottom=true, debug=false) {
  d = ifundef(dm[3], 0);
  p = pad_path(dm[0], dm[1], r=max(1.6, min(min(dm[0], dm[1])/2, r)));
  scatter(dm[0]+10, enable=!debug && lid && bin) {
    if (bin) difference() {
      fillet_bin(p, h=dm[2], t=t, bottom=bottom) if ($children>0) children(0);
    }
    if (lid) difference() {
      flipx(h=dm[2]+d+t*2+0.1, enable=debug) fillet_lid(p, h=d, t=t, e=e, g=g, top=top);
      if ($children>1) flipx(h=dm[2]+d+t*2+0.1, enable=!debug) ascend(t) children(1);
    }
  }
}

// weld children together with fillet edges, r=fillet radius, n=steps (could be slow if more than 3 children)
module weld(r=3, n) {
  /* private */ module _merge(r, n) {
    for(i=[1:n]) hull() {
      intersection() {
        children(0);
        minkowski() {
          children(2);
          render() sphere(r=r*i/n, $fn=12);
        }
      }
      intersection() {
        children(1);
        minkowski() {
          children(2);
          render() sphere(r=r*(n-i+1)/n, $fn=12);
        }
      }
    }
  }
  $fn = $fn ? $fn : $preview ? 16 : 0; // preview in low resolution if $fn not set
  children();
  if (r>0) for (i=[0:$children-1], j=[0:$children-1]) if (i<j) {
    _merge(r, $preview ? 1 : n==undef ? ceil(_fn(r)/4) : n) {
      children(i);
      children(j);
      render() intersection() {
        children(i);
        children(j);
      }
    }
  }
  if (!$preview) echo("welding... (may take a long while)");
}

// ====================================================================
// morph modules
// ====================================================================

// extrude vertically, morphing from p1 to p2, with height h and cyclic adjustment s
// guide is a 2D profile of control points, with each coordinate in [0,1] range
module morph_extrude(p1, p2, h, guide, s=0, curved=false) {
  n = max(len(p1), len(p2));
  q1 = cyclic(resample(p1, n), s);
  q2 = resample(p2, n);
  if (is_list(guide))
    layered_block([for (g=guide) force3d(morph(q1, q2, g[1])*g[0], g[1]*h)]);
  else
    layered_block([for (t=quanta(h/2, end=1)) force3d(morph(q1, q2, t, curved?t:1), t*h)]);
}

// extrude a dome morphing from p1 to p2, with height h and cyclic adjustment s
// guide is a 2D curve of control points, with each coordinate within the [0,1] range
module morph_dome(p1, p2, h, guide, s=0, inner=2) {
  p2 = ifundef(p2, p1);
  n = max(len(p1), len(p2));
  q1 = cyclic(resample(p1, n), s);
  q2 = resample(p2, n);
  hi = h - inner/2;
  guide = (guide!=undef?guide:round_path(h, a=[0,90])/h);
  layered_block(concat(
    [for (g=reverse(guide)) force3d(offset2d(morph(q1, q2, g[1])*g[0], -inner), g[1]*hi)],
    [for (g=guide) force3d(morph(q1, q2, g[1])*g[0], g[1]*h)]
  ));
}

// extrude a tray morphing from p1 to p2, with height h and cyclic adjustment s
// guide is a 2D curve of control points, with each coordinate within the [0,1] range
module morph_tray(p1, p2, h, guide, s=0, inner=2) {
  n = max(len(p1), len(p2));
  q1 = cyclic(resample(p1, n), s);
  q2 = resample(p2, n);
  hi = h - inner;
  guide = (guide!=undef?guide:[for (t=quanta(_fn(h), end=1)) [1,t]]);
  layered_block(concat(
    [for (g=(guide)) force3d(morph(q1, q2, g[1])*g[0], g[1]*h)],
    [for (g=reverse(guide)) force3d(offset2d(morph(q1, q2, g[1])*g[0], -inner), g[1]*hi+inner)]
  ));
}

// extrude a vase morphing from p1 to p2, with height h and cyclic adjustment s
// guide is a 2D curve of control points, with each coordinate within the [0,1] range
module morph_vase(p1, p2, h, guide, s=0, inner=2, tidy=0) {
  n = max(len(p1), len(p2));
  q1 = cyclic(resample(p1, n), s);
  q2 = resample(p2, n)*0.7;
  hi = h - inner/2;
  guide = (guide!=undef?guide:subarray(round_path(h, a=[90,0]), 1)/h);
  layered_block(concat(
    [for (g=snip(reverse(guide))) force3d(morph(q1, q2, g[1])*(1.5-g[0]), g[1]*h)],
    [for (g=guide) force3d(offset2d(morph(q1, q2, g[1])*(1.5-g[0]), -inner, tidy=tidy), g[1]*hi+inner)]
  ));
}

// create a complete case by morphing upward from profile p1 to p2, along the guide curve
// guide is defaulted to a dome shape if not supplied
module morph_case(p1, p2, h, guide, cut=0.5, s=0, j=5, inner=0.95, visible=true) {
  n = max(len(p1), len(p2));
  q1 = cyclic(resample(p1, n), s);
  q2 = resample(p2, n);
  guide = (guide!=undef?guide:shift2d(concat([for (t=quanta(20)) [1,-0.5*(1-t)]], scale2d(round_path(h, a=[0,90])/h, [1,0.5])), [0,0.5]));
  gap = 0.4/h;
  w = box2dw(p1)+5;

  // bin
  h1 = frame2d(guide, 0, 1, 0, cut);
  h2 = frame2d(guide*inner, 0, 1, cut, cut+j/h);
  e1 = last(h1);
  e2 = last(h2);
  h3 = reverse(frame2d(guide*inner*inner, 0, 1, 1-inner, e2[1]));
  g1 = concat(h1, [[h2[0][0], e1[1]]], shift2d(h2, [-gap,0]), [[h3[0][0], e2[1]]], h3);
  if (visible) translate([w/2,0,0]) morph_extrude(p1, p2, h, g1);

  // lid
  h4 = reverse(frame2d(guide, 0, 1, cut, 1));
  h5 = frame2d(guide*inner, 0, 1, cut, 1);
  g2 = concat(h4, [e1, [h2[0][0], e1[1]]], h5);
  translate([-w/2,0,-e1[1]*h]) {
    if (visible) morph_extrude(p1, p2, h, reverse(g2));
    children();
  }
}

// ====================================================================
// snap-fit box
// ====================================================================

module box_lid(dm, t=2, r=2, gh=3, gd=0.5, sp=0.2, hover=true) {
  ts = max(1.2, t/2);
  minor = min(dm[0], dm[1])/2;
  rr = confine(r-t, 1, minor-t-sp);
  flipx(a=hover ? 0 : 180, h=hover ? 0 : t+dm[2]+0.2) {
    ascend(dm[2]+0.1) {
      solid(pad_path(dm[0], dm[1], min(minor-sp, r)), t);
      ascend(-gh) shell(pad_path(dm[0]-t*2-sp, dm[1]-t*2-sp, 0), gh, -ts, r=rr);
    }
    if (gd) ascend(1-gh) box_gasket([dm[0]-t*2-sp, dm[1]-t*2-sp, dm[2]], d=gd+sp/2, r=rr);
    ascend(t-0.01) children();
  }
}

module box_bin(dm, t=2, r=2, gh=3, gd=0.5, sp=0.2, notch=false) {
  minor = min(dm[0], dm[1])/2;
  ns = min(5, dm[0]/7);
  rr = confine(r-t, 0.5, minor-t-sp);
  difference() {
    basin(quad_path(dm[0], dm[1]), dm[2], -t, r=min(minor-sp, r));
    if (gd) ascend(1-gh) box_gasket([dm[0]-t*2, dm[1]-t*2, dm[2]], d=gd+sp, r=rr);
    if (notch) mirror_clone() beam([-ns,-dm[1]/2,dm[2]], [ns,-dm[1]/2,dm[2]], min(3, t));
  }
  ascend(t-0.01) children();
}

module box_gasket(dm, d=1, r=2) {
  sweep(ring_path(d), force3d(pad_path(dm[0], dm[1], r), dm[2]), loop=true);
}
