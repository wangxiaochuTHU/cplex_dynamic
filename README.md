### cplex_dynamic
Binding of IBM Cplex dynamic/shared library, modified based on `rplex`.

### steps to use
1. install cplex on your computer. 
2. copy `..somepath\IBM\ILOG\CPLEX_Studio_XXXX\cplex\bin\x64_win64\cplexXXXX.dll` file (or `.so` in linux os) into your crate root folder (or its subfolder).

### check example
```
cargo run --example qcp
```


