
var ortWasm = (function() {
  var _scriptDir = typeof document !== 'undefined' && document.currentScript ? document.currentScript.src : undefined;
  if (typeof __filename !== 'undefined') _scriptDir = _scriptDir || __filename;
  return (
function(ortWasm) {
  ortWasm = ortWasm || {};


var e;e||(e=typeof ortWasm !== 'undefined' ? ortWasm : {});var aa,ba;e.ready=new Promise(function(a,b){aa=a;ba=b});var k={},u;for(u in e)e.hasOwnProperty(u)&&(k[u]=e[u]);var ca="./this.program",da="object"===typeof window,v="function"===typeof importScripts,ea="object"===typeof process&&"object"===typeof process.versions&&"string"===typeof process.versions.node,x="",fa,y,z,A,B;
if(ea)x=v?require("path").dirname(x)+"/":__dirname+"/",fa=function(a,b){A||(A=require("fs"));B||(B=require("path"));a=B.normalize(a);return A.readFileSync(a,b?null:"utf8")},z=function(a){a=fa(a,!0);a.buffer||(a=new Uint8Array(a));a.buffer||C("Assertion failed: undefined");return a},y=function(a,b,c){A||(A=require("fs"));B||(B=require("path"));a=B.normalize(a);A.readFile(a,function(d,f){d?c(d):b(f.buffer)})},1<process.argv.length&&(ca=process.argv[1].replace(/\\/g,"/")),process.argv.slice(2),process.on("uncaughtException",
function(a){throw a;}),process.on("unhandledRejection",C),e.inspect=function(){return"[Emscripten Module object]"};else if(da||v)v?x=self.location.href:"undefined"!==typeof document&&document.currentScript&&(x=document.currentScript.src),_scriptDir&&(x=_scriptDir),0!==x.indexOf("blob:")?x=x.substr(0,x.lastIndexOf("/")+1):x="",fa=function(a){var b=new XMLHttpRequest;b.open("GET",a,!1);b.send(null);return b.responseText},v&&(z=function(a){var b=new XMLHttpRequest;b.open("GET",a,!1);b.responseType="arraybuffer";
b.send(null);return new Uint8Array(b.response)}),y=function(a,b,c){var d=new XMLHttpRequest;d.open("GET",a,!0);d.responseType="arraybuffer";d.onload=function(){200==d.status||0==d.status&&d.response?b(d.response):c()};d.onerror=c;d.send(null)};var ha=e.print||console.log.bind(console),D=e.printErr||console.warn.bind(console);for(u in k)k.hasOwnProperty(u)&&(e[u]=k[u]);k=null;e.thisProgram&&(ca=e.thisProgram);var E;e.wasmBinary&&(E=e.wasmBinary);var noExitRuntime=e.noExitRuntime||!1;
"object"!==typeof WebAssembly&&C("no native wasm support detected");var ia,ja=!1,ka="undefined"!==typeof TextDecoder?new TextDecoder("utf8"):void 0;
function la(a,b,c){var d=b+c;for(c=b;a[c]&&!(c>=d);)++c;if(16<c-b&&a.subarray&&ka)return ka.decode(a.subarray(b,c));for(d="";b<c;){var f=a[b++];if(f&128){var h=a[b++]&63;if(192==(f&224))d+=String.fromCharCode((f&31)<<6|h);else{var l=a[b++]&63;f=224==(f&240)?(f&15)<<12|h<<6|l:(f&7)<<18|h<<12|l<<6|a[b++]&63;65536>f?d+=String.fromCharCode(f):(f-=65536,d+=String.fromCharCode(55296|f>>10,56320|f&1023))}}else d+=String.fromCharCode(f)}return d}function F(a,b){return a?la(H,a,b):""}
function ma(a,b,c,d){if(!(0<d))return 0;var f=c;d=c+d-1;for(var h=0;h<a.length;++h){var l=a.charCodeAt(h);if(55296<=l&&57343>=l){var m=a.charCodeAt(++h);l=65536+((l&1023)<<10)|m&1023}if(127>=l){if(c>=d)break;b[c++]=l}else{if(2047>=l){if(c+1>=d)break;b[c++]=192|l>>6}else{if(65535>=l){if(c+2>=d)break;b[c++]=224|l>>12}else{if(c+3>=d)break;b[c++]=240|l>>18;b[c++]=128|l>>12&63}b[c++]=128|l>>6&63}b[c++]=128|l&63}}b[c]=0;return c-f}function na(a,b,c){return ma(a,H,b,c)}
function oa(a){for(var b=0,c=0;c<a.length;++c){var d=a.charCodeAt(c);55296<=d&&57343>=d&&(d=65536+((d&1023)<<10)|a.charCodeAt(++c)&1023);127>=d?++b:b=2047>=d?b+2:65535>=d?b+3:b+4}return b}var pa="undefined"!==typeof TextDecoder?new TextDecoder("utf-16le"):void 0;function qa(a,b){var c=a>>1;for(var d=c+b/2;!(c>=d)&&ra[c];)++c;c<<=1;if(32<c-a&&pa)return pa.decode(H.subarray(a,c));c="";for(d=0;!(d>=b/2);++d){var f=I[a+2*d>>1];if(0==f)break;c+=String.fromCharCode(f)}return c}
function sa(a,b,c){void 0===c&&(c=2147483647);if(2>c)return 0;c-=2;var d=b;c=c<2*a.length?c/2:a.length;for(var f=0;f<c;++f)I[b>>1]=a.charCodeAt(f),b+=2;I[b>>1]=0;return b-d}function ta(a){return 2*a.length}function ua(a,b){for(var c=0,d="";!(c>=b/4);){var f=J[a+4*c>>2];if(0==f)break;++c;65536<=f?(f-=65536,d+=String.fromCharCode(55296|f>>10,56320|f&1023)):d+=String.fromCharCode(f)}return d}
function va(a,b,c){void 0===c&&(c=2147483647);if(4>c)return 0;var d=b;c=d+c-4;for(var f=0;f<a.length;++f){var h=a.charCodeAt(f);if(55296<=h&&57343>=h){var l=a.charCodeAt(++f);h=65536+((h&1023)<<10)|l&1023}J[b>>2]=h;b+=4;if(b+4>c)break}J[b>>2]=0;return b-d}function wa(a){for(var b=0,c=0;c<a.length;++c){var d=a.charCodeAt(c);55296<=d&&57343>=d&&++c;b+=4}return b}function xa(a){var b=oa(a)+1,c=K(b);c&&ma(a,L,c,b);return c}var ya,L,H,I,ra,J,N,za,Aa;
function Ba(){var a=ia.buffer;ya=a;e.HEAP8=L=new Int8Array(a);e.HEAP16=I=new Int16Array(a);e.HEAP32=J=new Int32Array(a);e.HEAPU8=H=new Uint8Array(a);e.HEAPU16=ra=new Uint16Array(a);e.HEAPU32=N=new Uint32Array(a);e.HEAPF32=za=new Float32Array(a);e.HEAPF64=Aa=new Float64Array(a)}var Ca,Da=[],Ea=[],Fa=[],Ga=[];function Ha(){var a=e.preRun.shift();Da.unshift(a)}var O=0,Ia=null,P=null;e.preloadedImages={};e.preloadedAudios={};
function C(a){if(e.onAbort)e.onAbort(a);D(a);ja=!0;a=new WebAssembly.RuntimeError("abort("+a+"). Build with -s ASSERTIONS=1 for more info.");ba(a);throw a;}function Ja(){return Q.startsWith("data:application/octet-stream;base64,")}var Q;Q="ort-wasm.wasm";if(!Ja()){var Ka=Q;Q=e.locateFile?e.locateFile(Ka,x):x+Ka}function La(){var a=Q;try{if(a==Q&&E)return new Uint8Array(E);if(z)return z(a);throw"both async and sync fetching of the wasm failed";}catch(b){C(b)}}
function Ma(){if(!E&&(da||v)){if("function"===typeof fetch&&!Q.startsWith("file://"))return fetch(Q,{credentials:"same-origin"}).then(function(a){if(!a.ok)throw"failed to load wasm binary file at '"+Q+"'";return a.arrayBuffer()}).catch(function(){return La()});if(y)return new Promise(function(a,b){y(Q,function(c){a(new Uint8Array(c))},b)})}return Promise.resolve().then(function(){return La()})}
function Na(a){for(;0<a.length;){var b=a.shift();if("function"==typeof b)b(e);else{var c=b.ib;"number"===typeof c?void 0===b.Za?Ca.get(c)():Ca.get(c)(b.Za):c(void 0===b.Za?null:b.Za)}}}function Oa(a){this.ab=a-16;this.tb=function(b){J[this.ab+4>>2]=b};this.qb=function(b){J[this.ab+8>>2]=b};this.rb=function(){J[this.ab>>2]=0};this.pb=function(){L[this.ab+12>>0]=0};this.sb=function(){L[this.ab+13>>0]=0};this.lb=function(b,c){this.tb(b);this.qb(c);this.rb();this.pb();this.sb()}}
var Pa=0,Qa={},Ra=[null,[],[]],R={};function Sa(a){switch(a){case 1:return 0;case 2:return 1;case 4:return 2;case 8:return 3;default:throw new TypeError("Unknown type size: "+a);}}var Ta=void 0;function S(a){for(var b="";H[a];)b+=Ta[H[a++]];return b}var Ua={},Va={},Wa={};function Xa(a){if(void 0===a)return"_unknown";a=a.replace(/[^a-zA-Z0-9_]/g,"$");var b=a.charCodeAt(0);return 48<=b&&57>=b?"_"+a:a}
function Ya(a,b){a=Xa(a);return(new Function("body","return function "+a+'() {\n    "use strict";    return body.apply(this, arguments);\n};\n'))(b)}function Za(a){var b=Error,c=Ya(a,function(d){this.name=a;this.message=d;d=Error(d).stack;void 0!==d&&(this.stack=this.toString()+"\n"+d.replace(/^Error(:[^\n]*)?\n/,""))});c.prototype=Object.create(b.prototype);c.prototype.constructor=c;c.prototype.toString=function(){return void 0===this.message?this.name:this.name+": "+this.message};return c}
var $a=void 0;function T(a){throw new $a(a);}function U(a,b,c){c=c||{};if(!("argPackAdvance"in b))throw new TypeError("registerType registeredInstance requires argPackAdvance");var d=b.name;a||T('type "'+d+'" must have a positive integer typeid pointer');if(Va.hasOwnProperty(a)){if(c.kb)return;T("Cannot register type '"+d+"' twice")}Va[a]=b;delete Wa[a];Ua.hasOwnProperty(a)&&(b=Ua[a],delete Ua[a],b.forEach(function(f){f()}))}var ab=[],V=[{},{value:void 0},{value:null},{value:!0},{value:!1}];
function bb(a){4<a&&0===--V[a].gb&&(V[a]=void 0,ab.push(a))}function W(a){switch(a){case void 0:return 1;case null:return 2;case !0:return 3;case !1:return 4;default:var b=ab.length?ab.pop():V.length;V[b]={gb:1,value:a};return b}}function cb(a){return this.fromWireType(N[a>>2])}function db(a){if(null===a)return"null";var b=typeof a;return"object"===b||"array"===b||"function"===b?a.toString():""+a}
function eb(a,b){switch(b){case 2:return function(c){return this.fromWireType(za[c>>2])};case 3:return function(c){return this.fromWireType(Aa[c>>3])};default:throw new TypeError("Unknown float type: "+a);}}
function fb(a,b,c){switch(b){case 0:return c?function(d){return L[d]}:function(d){return H[d]};case 1:return c?function(d){return I[d>>1]}:function(d){return ra[d>>1]};case 2:return c?function(d){return J[d>>2]}:function(d){return N[d>>2]};default:throw new TypeError("Unknown integer type: "+a);}}function X(a){a||T("Cannot use deleted val. handle = "+a);return V[a].value}function gb(a,b){var c=Va[a];if(void 0===c){a=hb(a);var d=S(a);Y(a);T(b+" has unknown type "+d)}return c}var ib={};
function jb(a){var b=ib[a];return void 0===b?S(a):b}var kb=[];function lb(){return"object"===typeof globalThis?globalThis:Function("return this")()}function mb(a){var b=kb.length;kb.push(a);return b}function nb(a,b){for(var c=Array(a),d=0;d<a;++d)c[d]=gb(J[(b>>2)+d],"parameter "+d);return c}
function ob(a){var b=Function;if(!(b instanceof Function))throw new TypeError("new_ called with constructor type "+typeof b+" which is not a function");var c=Ya(b.name||"unknownFunctionName",function(){});c.prototype=b.prototype;c=new c;a=b.apply(c,a);return a instanceof Object?a:c}var qb={},rb;rb=ea?function(){var a=process.hrtime();return 1E3*a[0]+a[1]/1E6}:function(){return performance.now()};var sb={};
function tb(){if(!ub){var a={USER:"web_user",LOGNAME:"web_user",PATH:"/",PWD:"/",HOME:"/home/web_user",LANG:("object"===typeof navigator&&navigator.languages&&navigator.languages[0]||"C").replace("-","_")+".UTF-8",_:ca||"./this.program"},b;for(b in sb)void 0===sb[b]?delete a[b]:a[b]=sb[b];var c=[];for(b in a)c.push(b+"="+a[b]);ub=c}return ub}var ub;
function vb(a,b){a=new Date(1E3*J[a>>2]);J[b>>2]=a.getUTCSeconds();J[b+4>>2]=a.getUTCMinutes();J[b+8>>2]=a.getUTCHours();J[b+12>>2]=a.getUTCDate();J[b+16>>2]=a.getUTCMonth();J[b+20>>2]=a.getUTCFullYear()-1900;J[b+24>>2]=a.getUTCDay();J[b+36>>2]=0;J[b+32>>2]=0;J[b+28>>2]=(a.getTime()-Date.UTC(a.getUTCFullYear(),0,1,0,0,0,0))/864E5|0;vb.hb||(vb.hb=xa("GMT"));J[b+40>>2]=vb.hb;return b}
function wb(){function a(l){return(l=l.toTimeString().match(/\(([A-Za-z ]+)\)$/))?l[1]:"GMT"}if(!xb){xb=!0;var b=(new Date).getFullYear(),c=new Date(b,0,1),d=new Date(b,6,1);b=c.getTimezoneOffset();var f=d.getTimezoneOffset(),h=Math.max(b,f);J[yb()>>2]=60*h;J[zb()>>2]=Number(b!=f);c=a(c);d=a(d);c=xa(c);d=xa(d);f<b?(J[Z()>>2]=c,J[Z()+4>>2]=d):(J[Z()>>2]=d,J[Z()+4>>2]=c)}}var xb;function Ab(a){return 0===a%4&&(0!==a%100||0===a%400)}function Bb(a,b){for(var c=0,d=0;d<=b;c+=a[d++]);return c}
var Cb=[31,29,31,30,31,30,31,31,30,31,30,31],Db=[31,28,31,30,31,30,31,31,30,31,30,31];function Eb(a,b){for(a=new Date(a.getTime());0<b;){var c=a.getMonth(),d=(Ab(a.getFullYear())?Cb:Db)[c];if(b>d-a.getDate())b-=d-a.getDate()+1,a.setDate(1),11>c?a.setMonth(c+1):(a.setMonth(0),a.setFullYear(a.getFullYear()+1));else{a.setDate(a.getDate()+b);break}}return a}
function Fb(a,b,c,d){function f(g,n,t){for(g="number"===typeof g?g.toString():g||"";g.length<n;)g=t[0]+g;return g}function h(g,n){return f(g,n,"0")}function l(g,n){function t(pb){return 0>pb?-1:0<pb?1:0}var M;0===(M=t(g.getFullYear()-n.getFullYear()))&&0===(M=t(g.getMonth()-n.getMonth()))&&(M=t(g.getDate()-n.getDate()));return M}function m(g){switch(g.getDay()){case 0:return new Date(g.getFullYear()-1,11,29);case 1:return g;case 2:return new Date(g.getFullYear(),0,3);case 3:return new Date(g.getFullYear(),
0,2);case 4:return new Date(g.getFullYear(),0,1);case 5:return new Date(g.getFullYear()-1,11,31);case 6:return new Date(g.getFullYear()-1,11,30)}}function p(g){g=Eb(new Date(g.Xa+1900,0,1),g.fb);var n=new Date(g.getFullYear()+1,0,4),t=m(new Date(g.getFullYear(),0,4));n=m(n);return 0>=l(t,g)?0>=l(n,g)?g.getFullYear()+1:g.getFullYear():g.getFullYear()-1}var r=J[d+40>>2];d={wb:J[d>>2],vb:J[d+4>>2],cb:J[d+8>>2],bb:J[d+12>>2],Ya:J[d+16>>2],Xa:J[d+20>>2],eb:J[d+24>>2],fb:J[d+28>>2],Eb:J[d+32>>2],ub:J[d+
36>>2],xb:r?F(r):""};c=F(c);r={"%c":"%a %b %d %H:%M:%S %Y","%D":"%m/%d/%y","%F":"%Y-%m-%d","%h":"%b","%r":"%I:%M:%S %p","%R":"%H:%M","%T":"%H:%M:%S","%x":"%m/%d/%y","%X":"%H:%M:%S","%Ec":"%c","%EC":"%C","%Ex":"%m/%d/%y","%EX":"%H:%M:%S","%Ey":"%y","%EY":"%Y","%Od":"%d","%Oe":"%e","%OH":"%H","%OI":"%I","%Om":"%m","%OM":"%M","%OS":"%S","%Ou":"%u","%OU":"%U","%OV":"%V","%Ow":"%w","%OW":"%W","%Oy":"%y"};for(var q in r)c=c.replace(new RegExp(q,"g"),r[q]);var w="Sunday Monday Tuesday Wednesday Thursday Friday Saturday".split(" "),
G="January February March April May June July August September October November December".split(" ");r={"%a":function(g){return w[g.eb].substring(0,3)},"%A":function(g){return w[g.eb]},"%b":function(g){return G[g.Ya].substring(0,3)},"%B":function(g){return G[g.Ya]},"%C":function(g){return h((g.Xa+1900)/100|0,2)},"%d":function(g){return h(g.bb,2)},"%e":function(g){return f(g.bb,2," ")},"%g":function(g){return p(g).toString().substring(2)},"%G":function(g){return p(g)},"%H":function(g){return h(g.cb,
2)},"%I":function(g){g=g.cb;0==g?g=12:12<g&&(g-=12);return h(g,2)},"%j":function(g){return h(g.bb+Bb(Ab(g.Xa+1900)?Cb:Db,g.Ya-1),3)},"%m":function(g){return h(g.Ya+1,2)},"%M":function(g){return h(g.vb,2)},"%n":function(){return"\n"},"%p":function(g){return 0<=g.cb&&12>g.cb?"AM":"PM"},"%S":function(g){return h(g.wb,2)},"%t":function(){return"\t"},"%u":function(g){return g.eb||7},"%U":function(g){var n=new Date(g.Xa+1900,0,1),t=0===n.getDay()?n:Eb(n,7-n.getDay());g=new Date(g.Xa+1900,g.Ya,g.bb);return 0>
l(t,g)?h(Math.ceil((31-t.getDate()+(Bb(Ab(g.getFullYear())?Cb:Db,g.getMonth()-1)-31)+g.getDate())/7),2):0===l(t,n)?"01":"00"},"%V":function(g){var n=new Date(g.Xa+1901,0,4),t=m(new Date(g.Xa+1900,0,4));n=m(n);var M=Eb(new Date(g.Xa+1900,0,1),g.fb);return 0>l(M,t)?"53":0>=l(n,M)?"01":h(Math.ceil((t.getFullYear()<g.Xa+1900?g.fb+32-t.getDate():g.fb+1-t.getDate())/7),2)},"%w":function(g){return g.eb},"%W":function(g){var n=new Date(g.Xa,0,1),t=1===n.getDay()?n:Eb(n,0===n.getDay()?1:7-n.getDay()+1);g=
new Date(g.Xa+1900,g.Ya,g.bb);return 0>l(t,g)?h(Math.ceil((31-t.getDate()+(Bb(Ab(g.getFullYear())?Cb:Db,g.getMonth()-1)-31)+g.getDate())/7),2):0===l(t,n)?"01":"00"},"%y":function(g){return(g.Xa+1900).toString().substring(2)},"%Y":function(g){return g.Xa+1900},"%z":function(g){g=g.ub;var n=0<=g;g=Math.abs(g)/60;return(n?"+":"-")+String("0000"+(g/60*100+g%60)).slice(-4)},"%Z":function(g){return g.xb},"%%":function(){return"%"}};for(q in r)c.includes(q)&&(c=c.replace(new RegExp(q,"g"),r[q](d)));q=Gb(c);
if(q.length>b)return 0;L.set(q,a);return q.length-1}for(var Hb=Array(256),Ib=0;256>Ib;++Ib)Hb[Ib]=String.fromCharCode(Ib);Ta=Hb;$a=e.BindingError=Za("BindingError");e.InternalError=Za("InternalError");e.count_emval_handles=function(){for(var a=0,b=5;b<V.length;++b)void 0!==V[b]&&++a;return a};e.get_first_emval=function(){for(var a=5;a<V.length;++a)if(void 0!==V[a])return V[a];return null};function Gb(a){var b=Array(oa(a)+1);ma(a,b,0,b.length);return b}
var Lb={a:function(a){return K(a+16)+16},c:function(a,b){Fa.unshift({ib:a,Za:b})},k:function(a,b){Fa.unshift({ib:a,Za:b})},b:function(a,b,c){(new Oa(a)).lb(b,c);Pa++;throw a;},X:function(a,b){a=F(a);return R.yb(a,b)},C:function(){return 0},aa:function(){},da:function(){},E:function(){return 42},Q:function(){return 0},$:function(){},_:function(a,b){a=F(a);return R.zb(a,b)},ca:function(a,b,c,d,f,h){h<<=12;if(0!==(d&16)&&0!==a%65536)b=-28;else if(0!==(d&32)){a=65536*Math.ceil(b/65536);var l=Jb(65536,
a);l?(H.fill(0,l,l+a),a=l):a=0;a?(Qa[a]={ob:a,nb:b,jb:!0,fd:f,Db:c,flags:d,offset:h},b=a):b=-48}else b=-52;return b},ba:function(a,b){var c=Qa[a];0!==b&&c?(b===c.nb&&(Qa[a]=null,c.jb&&Y(c.ob)),a=0):a=-28;return a},y:function(){},W:function(a,b,c){a=F(a);return R.Ab(a,b,c)},Y:function(){},H:function(){},Z:function(){},O:function(){},ha:function(a,b,c,d,f){var h=Sa(c);b=S(b);U(a,{name:b,fromWireType:function(l){return!!l},toWireType:function(l,m){return m?d:f},argPackAdvance:8,readValueFromPointer:function(l){if(1===
c)var m=L;else if(2===c)m=I;else if(4===c)m=J;else throw new TypeError("Unknown boolean type size: "+b);return this.fromWireType(m[l>>h])},$a:null})},ga:function(a,b){b=S(b);U(a,{name:b,fromWireType:function(c){var d=V[c].value;bb(c);return d},toWireType:function(c,d){return W(d)},argPackAdvance:8,readValueFromPointer:cb,$a:null})},I:function(a,b,c){c=Sa(c);b=S(b);U(a,{name:b,fromWireType:function(d){return d},toWireType:function(d,f){if("number"!==typeof f&&"boolean"!==typeof f)throw new TypeError('Cannot convert "'+
db(f)+'" to '+this.name);return f},argPackAdvance:8,readValueFromPointer:eb(b,c),$a:null})},r:function(a,b,c,d,f){function h(r){return r}b=S(b);-1===f&&(f=4294967295);var l=Sa(c);if(0===d){var m=32-8*c;h=function(r){return r<<m>>>m}}var p=b.includes("unsigned");U(a,{name:b,fromWireType:h,toWireType:function(r,q){if("number"!==typeof q&&"boolean"!==typeof q)throw new TypeError('Cannot convert "'+db(q)+'" to '+this.name);if(q<d||q>f)throw new TypeError('Passing a number "'+db(q)+'" from JS side to C/C++ side to an argument of type "'+
b+'", which is outside the valid range ['+d+", "+f+"]!");return p?q>>>0:q|0},argPackAdvance:8,readValueFromPointer:fb(b,l,0!==d),$a:null})},q:function(a,b,c){function d(h){h>>=2;var l=N;return new f(ya,l[h+1],l[h])}var f=[Int8Array,Uint8Array,Int16Array,Uint16Array,Int32Array,Uint32Array,Float32Array,Float64Array][b];c=S(c);U(a,{name:c,fromWireType:d,argPackAdvance:8,readValueFromPointer:d},{kb:!0})},J:function(a,b){b=S(b);var c="std::string"===b;U(a,{name:b,fromWireType:function(d){var f=N[d>>2];
if(c)for(var h=d+4,l=0;l<=f;++l){var m=d+4+l;if(l==f||0==H[m]){h=F(h,m-h);if(void 0===p)var p=h;else p+=String.fromCharCode(0),p+=h;h=m+1}}else{p=Array(f);for(l=0;l<f;++l)p[l]=String.fromCharCode(H[d+4+l]);p=p.join("")}Y(d);return p},toWireType:function(d,f){f instanceof ArrayBuffer&&(f=new Uint8Array(f));var h="string"===typeof f;h||f instanceof Uint8Array||f instanceof Uint8ClampedArray||f instanceof Int8Array||T("Cannot pass non-string to std::string");var l=(c&&h?function(){return oa(f)}:function(){return f.length})(),
m=K(4+l+1);N[m>>2]=l;if(c&&h)na(f,m+4,l+1);else if(h)for(h=0;h<l;++h){var p=f.charCodeAt(h);255<p&&(Y(m),T("String has UTF-16 code units that do not fit in 8 bits"));H[m+4+h]=p}else for(h=0;h<l;++h)H[m+4+h]=f[h];null!==d&&d.push(Y,m);return m},argPackAdvance:8,readValueFromPointer:cb,$a:function(d){Y(d)}})},z:function(a,b,c){c=S(c);if(2===b){var d=qa;var f=sa;var h=ta;var l=function(){return ra};var m=1}else 4===b&&(d=ua,f=va,h=wa,l=function(){return N},m=2);U(a,{name:c,fromWireType:function(p){for(var r=
N[p>>2],q=l(),w,G=p+4,g=0;g<=r;++g){var n=p+4+g*b;if(g==r||0==q[n>>m])G=d(G,n-G),void 0===w?w=G:(w+=String.fromCharCode(0),w+=G),G=n+b}Y(p);return w},toWireType:function(p,r){"string"!==typeof r&&T("Cannot pass non-string to C++ string type "+c);var q=h(r),w=K(4+q+b);N[w>>2]=q>>m;f(r,w+4,q+b);null!==p&&p.push(Y,w);return w},argPackAdvance:8,readValueFromPointer:cb,$a:function(p){Y(p)}})},ia:function(a,b){b=S(b);U(a,{mb:!0,name:b,argPackAdvance:0,fromWireType:function(){},toWireType:function(){}})},
A:function(a,b,c){a=X(a);b=gb(b,"emval::as");var d=[],f=W(d);J[c>>2]=f;return b.toWireType(d,a)},p:function(a,b,c,d,f){a=kb[a];b=X(b);c=jb(c);var h=[];J[d>>2]=W(h);return a(b,c,h,f)},l:function(a,b,c,d){a=kb[a];b=X(b);c=jb(c);a(b,c,null,d)},d:bb,L:function(a,b){a=X(a);b=X(b);return a==b},w:function(a){if(0===a)return W(lb());a=jb(a);return W(lb()[a])},h:function(a,b){b=nb(a,b);for(var c=b[0],d=c.name+"_$"+b.slice(1).map(function(r){return r.name}).join("_")+"$",f=["retType"],h=[c],l="",m=0;m<a-1;++m)l+=
(0!==m?", ":"")+"arg"+m,f.push("argType"+m),h.push(b[1+m]);d="return function "+Xa("methodCaller_"+d)+"(handle, name, destructors, args) {\n";var p=0;for(m=0;m<a-1;++m)d+="    var arg"+m+" = argType"+m+".readValueFromPointer(args"+(p?"+"+p:"")+");\n",p+=b[m+1].argPackAdvance;d+="    var rv = handle[name]("+l+");\n";for(m=0;m<a-1;++m)b[m+1].deleteObject&&(d+="    argType"+m+".deleteObject(arg"+m+");\n");c.mb||(d+="    return retType.toWireType(destructors, rv);\n");f.push(d+"};\n");a=ob(f).apply(null,
h);return mb(a)},u:function(a,b){a=X(a);b=X(b);return W(a[b])},g:function(a){4<a&&(V[a].gb+=1)},ja:function(a,b,c,d){a=X(a);var f=qb[b];if(!f){f="";for(var h=0;h<b;++h)f+=(0!==h?", ":"")+"arg"+h;var l="return function emval_allocator_"+b+"(constructor, argTypes, args) {\n";for(h=0;h<b;++h)l+="var argType"+h+" = requireRegisteredType(Module['HEAP32'][(argTypes >>> 2) + "+h+'], "parameter '+h+'");\nvar arg'+h+" = argType"+h+".readValueFromPointer(args);\nargs += argType"+h+"['argPackAdvance'];\n";f=
(new Function("requireRegisteredType","Module","__emval_register",l+("var obj = new constructor("+f+");\nreturn __emval_register(obj);\n}\n")))(gb,e,W);qb[b]=f}return f(a,c,d)},m:function(){return W([])},e:function(a){return W(jb(a))},i:function(){return W({})},n:function(a){for(var b=V[a].value;b.length;){var c=b.pop();b.pop()(c)}bb(a)},f:function(a,b,c){a=X(a);b=X(b);c=X(c);a[b]=c},j:function(a,b){a=gb(a,"_emval_take_value");a=a.readValueFromPointer(b);return W(a)},v:function(){C()},K:function(a,
b){if(0===a)a=Date.now();else if(1===a||4===a)a=rb();else return J[Kb()>>2]=28,-1;J[b>>2]=a/1E3|0;J[b+4>>2]=a%1E3*1E6|0;return 0},M:function(a,b){return a-b},ma:function(){C("To use dlopen, you need to use Emscripten's linking support, see https://github.com/emscripten-core/emscripten/wiki/Linking")},t:function(){C("To use dlopen, you need to use Emscripten's linking support, see https://github.com/emscripten-core/emscripten/wiki/Linking")},na:function(){C("To use dlopen, you need to use Emscripten's linking support, see https://github.com/emscripten-core/emscripten/wiki/Linking")},
la:function(){C("To use dlopen, you need to use Emscripten's linking support, see https://github.com/emscripten-core/emscripten/wiki/Linking")},ea:function(){return 2147483648},P:function(a,b,c){H.copyWithin(a,b,b+c)},x:function(a){var b=H.length;a>>>=0;if(2147483648<a)return!1;for(var c=1;4>=c;c*=2){var d=b*(1+.2/c);d=Math.min(d,a+100663296);d=Math.max(a,d);0<d%65536&&(d+=65536-d%65536);a:{try{ia.grow(Math.min(2147483648,d)-ya.byteLength+65535>>>16);Ba();var f=1;break a}catch(h){}f=void 0}if(f)return!0}return!1},
V:function(a){for(var b=rb();rb()-b<a;);},T:function(a,b){var c=0;tb().forEach(function(d,f){var h=b+c;f=J[a+4*f>>2]=h;for(h=0;h<d.length;++h)L[f++>>0]=d.charCodeAt(h);L[f>>0]=0;c+=d.length+1});return 0},U:function(a,b){var c=tb();J[a>>2]=c.length;var d=0;c.forEach(function(f){d+=f.length+1});J[b>>2]=d;return 0},s:function(){return 0},R:function(a,b){a=1==a||2==a?2:C();L[b>>0]=a;return 0},D:function(a,b,c,d){a=R.Cb(a);b=R.Bb(a,b,c);J[d>>2]=b;return 0},N:function(){},F:function(a,b,c,d){for(var f=
0,h=0;h<c;h++){for(var l=J[b+8*h>>2],m=J[b+(8*h+4)>>2],p=0;p<m;p++){var r=H[l+p],q=Ra[a];0===r||10===r?((1===a?ha:D)(la(q,0)),q.length=0):q.push(r)}f+=m}J[d>>2]=f;return 0},fa:function(a){var b=Date.now();J[a>>2]=b/1E3|0;J[a+4>>2]=b%1E3*1E3|0;return 0},S:vb,G:function(a,b){wb();a=new Date(1E3*J[a>>2]);J[b>>2]=a.getSeconds();J[b+4>>2]=a.getMinutes();J[b+8>>2]=a.getHours();J[b+12>>2]=a.getDate();J[b+16>>2]=a.getMonth();J[b+20>>2]=a.getFullYear()-1900;J[b+24>>2]=a.getDay();var c=new Date(a.getFullYear(),
0,1);J[b+28>>2]=(a.getTime()-c.getTime())/864E5|0;J[b+36>>2]=-(60*a.getTimezoneOffset());var d=(new Date(a.getFullYear(),6,1)).getTimezoneOffset();c=c.getTimezoneOffset();a=(d!=c&&a.getTimezoneOffset()==Math.min(c,d))|0;J[b+32>>2]=a;a=J[Z()+(a?4:0)>>2];J[b+40>>2]=a;return b},B:function(a){wb();var b=new Date(J[a+20>>2]+1900,J[a+16>>2],J[a+12>>2],J[a+8>>2],J[a+4>>2],J[a>>2],0),c=J[a+32>>2],d=b.getTimezoneOffset(),f=new Date(b.getFullYear(),0,1),h=(new Date(b.getFullYear(),6,1)).getTimezoneOffset(),
l=f.getTimezoneOffset(),m=Math.min(l,h);0>c?J[a+32>>2]=Number(h!=l&&m==d):0<c!=(m==d)&&(h=Math.max(l,h),b.setTime(b.getTime()+6E4*((0<c?m:h)-d)));J[a+24>>2]=b.getDay();J[a+28>>2]=(b.getTime()-f.getTime())/864E5|0;J[a>>2]=b.getSeconds();J[a+4>>2]=b.getMinutes();J[a+8>>2]=b.getHours();J[a+12>>2]=b.getDate();J[a+16>>2]=b.getMonth();return b.getTime()/1E3|0},ka:Fb,o:function(a,b,c,d){return Fb(a,b,c,d)}};
(function(){function a(f){e.asm=f.exports;ia=e.asm.oa;Ba();Ca=e.asm.Wa;Ea.unshift(e.asm.pa);O--;e.monitorRunDependencies&&e.monitorRunDependencies(O);0==O&&(null!==Ia&&(clearInterval(Ia),Ia=null),P&&(f=P,P=null,f()))}function b(f){a(f.instance)}function c(f){return Ma().then(function(h){return WebAssembly.instantiate(h,d)}).then(f,function(h){D("failed to asynchronously prepare wasm: "+h);C(h)})}var d={a:Lb};O++;e.monitorRunDependencies&&e.monitorRunDependencies(O);if(e.instantiateWasm)try{return e.instantiateWasm(d,
a)}catch(f){return D("Module.instantiateWasm callback failed with error: "+f),!1}(function(){return E||"function"!==typeof WebAssembly.instantiateStreaming||Ja()||Q.startsWith("file://")||"function"!==typeof fetch?c(b):fetch(Q,{credentials:"same-origin"}).then(function(f){return WebAssembly.instantiateStreaming(f,d).then(b,function(h){D("wasm streaming compile failed: "+h);D("falling back to ArrayBuffer instantiation");return c(b)})})})().catch(ba);return{}})();
e.___wasm_call_ctors=function(){return(e.___wasm_call_ctors=e.asm.pa).apply(null,arguments)};e._OrtInit=function(){return(e._OrtInit=e.asm.qa).apply(null,arguments)};e._OrtCreateSessionOptions=function(){return(e._OrtCreateSessionOptions=e.asm.ra).apply(null,arguments)};e._OrtSessionOptionsAppendExecutionProviderWebNN=function(){return(e._OrtSessionOptionsAppendExecutionProviderWebNN=e.asm.sa).apply(null,arguments)};
e._OrtAddSessionConfigEntry=function(){return(e._OrtAddSessionConfigEntry=e.asm.ta).apply(null,arguments)};e._OrtReleaseSessionOptions=function(){return(e._OrtReleaseSessionOptions=e.asm.ua).apply(null,arguments)};e._OrtCreateSession=function(){return(e._OrtCreateSession=e.asm.va).apply(null,arguments)};e._OrtReleaseSession=function(){return(e._OrtReleaseSession=e.asm.wa).apply(null,arguments)};e._OrtGetInputCount=function(){return(e._OrtGetInputCount=e.asm.xa).apply(null,arguments)};
e._OrtGetOutputCount=function(){return(e._OrtGetOutputCount=e.asm.ya).apply(null,arguments)};e._OrtGetInputName=function(){return(e._OrtGetInputName=e.asm.za).apply(null,arguments)};e._OrtGetOutputName=function(){return(e._OrtGetOutputName=e.asm.Aa).apply(null,arguments)};e._OrtFree=function(){return(e._OrtFree=e.asm.Ba).apply(null,arguments)};e._OrtCreateTensor=function(){return(e._OrtCreateTensor=e.asm.Ca).apply(null,arguments)};
e._OrtGetTensorData=function(){return(e._OrtGetTensorData=e.asm.Da).apply(null,arguments)};e._OrtReleaseTensor=function(){return(e._OrtReleaseTensor=e.asm.Ea).apply(null,arguments)};e._OrtCreateRunOptions=function(){return(e._OrtCreateRunOptions=e.asm.Fa).apply(null,arguments)};e._OrtAddRunConfigEntry=function(){return(e._OrtAddRunConfigEntry=e.asm.Ga).apply(null,arguments)};e._OrtReleaseRunOptions=function(){return(e._OrtReleaseRunOptions=e.asm.Ha).apply(null,arguments)};
e._OrtRun=function(){return(e._OrtRun=e.asm.Ia).apply(null,arguments)};e._OrtEndProfiling=function(){return(e._OrtEndProfiling=e.asm.Ja).apply(null,arguments)};var K=e._malloc=function(){return(K=e._malloc=e.asm.Ka).apply(null,arguments)},Kb=e.___errno_location=function(){return(Kb=e.___errno_location=e.asm.La).apply(null,arguments)},Y=e._free=function(){return(Y=e._free=e.asm.Ma).apply(null,arguments)},hb=e.___getTypeName=function(){return(hb=e.___getTypeName=e.asm.Na).apply(null,arguments)};
e.___embind_register_native_and_builtin_types=function(){return(e.___embind_register_native_and_builtin_types=e.asm.Oa).apply(null,arguments)};
var Z=e.__get_tzname=function(){return(Z=e.__get_tzname=e.asm.Pa).apply(null,arguments)},zb=e.__get_daylight=function(){return(zb=e.__get_daylight=e.asm.Qa).apply(null,arguments)},yb=e.__get_timezone=function(){return(yb=e.__get_timezone=e.asm.Ra).apply(null,arguments)},Mb=e.stackSave=function(){return(Mb=e.stackSave=e.asm.Sa).apply(null,arguments)},Nb=e.stackRestore=function(){return(Nb=e.stackRestore=e.asm.Ta).apply(null,arguments)},Ob=e.stackAlloc=function(){return(Ob=e.stackAlloc=e.asm.Ua).apply(null,
arguments)},Jb=e._memalign=function(){return(Jb=e._memalign=e.asm.Va).apply(null,arguments)};e.UTF8ToString=F;e.stringToUTF8=na;e.lengthBytesUTF8=oa;e.stackSave=Mb;e.stackRestore=Nb;e.stackAlloc=Ob;var Pb;P=function Qb(){Pb||Rb();Pb||(P=Qb)};
function Rb(){function a(){if(!Pb&&(Pb=!0,e.calledRun=!0,!ja)){Na(Ea);aa(e);if(e.onRuntimeInitialized)e.onRuntimeInitialized();if(e.postRun)for("function"==typeof e.postRun&&(e.postRun=[e.postRun]);e.postRun.length;){var b=e.postRun.shift();Ga.unshift(b)}Na(Ga)}}if(!(0<O)){if(e.preRun)for("function"==typeof e.preRun&&(e.preRun=[e.preRun]);e.preRun.length;)Ha();Na(Da);0<O||(e.setStatus?(e.setStatus("Running..."),setTimeout(function(){setTimeout(function(){e.setStatus("")},1);a()},1)):a())}}e.run=Rb;
if(e.preInit)for("function"==typeof e.preInit&&(e.preInit=[e.preInit]);0<e.preInit.length;)e.preInit.pop()();Rb();


  return ortWasm.ready
}
);
})();
if (typeof exports === 'object' && typeof module === 'object')
  module.exports = ortWasm;
else if (typeof define === 'function' && define['amd'])
  define([], function() { return ortWasm; });
else if (typeof exports === 'object')
  exports["ortWasm"] = ortWasm;
