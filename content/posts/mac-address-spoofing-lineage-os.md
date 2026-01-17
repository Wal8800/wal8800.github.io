---
title: "Mac address spoofing in LineageOS 17.1 on Raspberry Pi 3"
date: 2022-02-28T20:15:48+13:00
draft: false
---

I wanted to change the mac address on the Raspberry Pi. After a bit of googling, I was able to change the mac address using `ip link` however
when the device is rebooted, the changes are reverted. It appears there isn't an easy way to change the mac address permanently so I decided to create
a script to apply the changes on start up.

Fortunately, the LineageOS raspberry build already came with support for running start up scripts from `/system/etc/init.d/`. This means I just needed to add script in there.

Here are steps I took:

1. Enable the developer options and enable the shell emulator.


2. Remount the file system with write permission if it doesn't have it already.


```sh
mount -oremount,rw /system
```

<ol start="3">
<li>Create a script in <code>/system/etc/init.d/</code>.</li>
</ol>

```sh
touch /system/etc/init.d/99macaddrchange
```

<ol start="4">
<li>Add the following content to the script.</li>
</ol>

```sh
#!/bin/sh
set -ex

# If "eth0" is the network adapter we want to change.
ip link set dev eth0 down
ip link set dev eth0 macaddr xx:xx:xx:xx:xx:xx
ip link set dev eth0 up
```

<ol start="5">
<li>Reboot the Raspberry Pi and run <code>ip link show</code> again to check if the changes are automatically applied.</li>
</ol>


References:

- [man page for ip link](https://man7.org/linux/man-pages/man8/ip-link.8.html)
- [Konstakang's LineageOS 17.1 build for Raspberry Pi 3](https://konstakang.com/devices/rpi3/LineageOS17.1/)
- [Guide to add start up scripts to LineageOS 16](https://h4des.org/blog/index.php?/archives/359-Android-LineageOS-16-Execute-Script-on-Start-Up.html)
