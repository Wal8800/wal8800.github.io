---
layout: post
title:  "Mac address spoofing in LineageOS 17.1 on Raspberry Pi 3"
date:   2022-02-28 20:15:48 +1300
categories: jekyll update
---

I wanted to change the mac address on the Raspberry Pi. After a bit of googling, I was able to change the mac address using `ip link` however
when the device is rebooted, the changes are reverted. It appears there isn't an easy way to change the mac address permanently so I decided to create 
a script to apply the changes on start up.

Fortunately, the LineageOS raspberry build already came with support for running start up scripts from `/system/etc/init.d/`. This means I just needed to add script in there.

Here are steps I took:

1. Enable the developer options and enable the shell emulator.


2. Remount the file system with write permission if it doesn't have it already.


{% highlight sh %}
mount -oremount,rw /system
{% endhighlight %}


{:start="3"}
3. Create a script in `/system/etc/init.d/`.

{% highlight sh %}
touch /system/etc/init.d/99macaddrchange
{% endhighlight %}

{:start="4"}
4. Add the following content to the script.

{% highlight sh %}
#!/bin/sh
set -ex

# If "eth0" is the network adapter we want to change.
ip link set dev eth0 down
ip link set dev eth0 macaddr xx:xx:xx:xx:xx:xx
ip link set dev eth0 up
{% endhighlight %}


{:start="5"}
5. Reboot the Raspberry Pi and run `ip link show` again to check if the changes are automatically applied.



References:

- [man page for ip link](https://man7.org/linux/man-pages/man8/ip-link.8.html){:target="_blank"}{:rel="noopener noreferrer"}
- [Konstakang's LineageOS 17.1 build for Raspberry Pi 3](https://konstakang.com/devices/rpi3/LineageOS17.1/){:target="_blank"}{:rel="noopener noreferrer"}
- [Guide to add start up scripts to LineageOS 16](https://h4des.org/blog/index.php?/archives/359-Android-LineageOS-16-Execute-Script-on-Start-Up.html){:target="_blank"}{:rel="noopener noreferrer"}

