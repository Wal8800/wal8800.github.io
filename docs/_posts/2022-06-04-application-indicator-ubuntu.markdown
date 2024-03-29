---
layout: post
title: "Application Indicator in Ubuntu 20.04"
date:   2022-06-04 17:47:48 +1300
--- 

At my previous job, I was using [this trailer app](http://ptsochantaris.github.io/trailer/) to access and manage my github pull requests. The trailer app provides an app indicator on top of the menu bar. Clicking on the icon shows a list of pull requests along their status and the list gets automatically updated. This app is only available on MacOS.

After switching to my new job, I started using Ubuntu 20.04 and I couldn't find an equivalent application. I wanted to see what it takes to create an app indicator application on Ubuntu and maybe create one myself if it's easy. 

A quick google search returns a [documentation](https://wiki.ubuntu.com/DesktopExperienceTeam/ApplicationIndicators) about application indicator in Ubuntu. The documentation was written some time ago (2013-04-11 13:15:38) but it is detailed and provided some example codes. I tried out the python example but the indicator icon didn't appear on the UI. After some more googling, it turns out for Ubuntu 20.04, I needed to install  [gnome-shell-extension-appindicator](https://github.com/ubuntu/gnome-shell-extension-appindicator) to display the app indicator.

With the extension installed, I ran the python example again and was able to get the indicator and menu to appear.

<div style="text-align: center; padding-bottom: 15px">
    <img src="{{ '/assets/img/appindicator/example_appindicator.png' | relative_url }}" />
</div>


I also tried to set one of the menu label with a markup style.

```python
    test_item = Gtk.MenuItem("")
    label = test_item.get_children()[0]
    label.set_markup("<b>Click here</b>")
```

Interestingly, the text wasn't applied with the style. 

<div style="text-align: center; padding-bottom: 15px">
    <img src="{{ '/assets/img/appindicator/no_styling_appindicator.png' | relative_url }}" />
</div>

At this point, I didn't know how the app indicator work to determine whether I could create something similar to the trailer app. Questions that I want to figure out was:

- Why do we need the additional extension to display the app indicator icon and menu?
- Why did the markup styling didn't apply on the app indicator menu label?

So I started digging into the details and noted my findings in the rest of this post. 


## Displaying the app indicator menu

Re-reading the official documentation, particularly on the "Software Architecture" section, I begin to see why the extension component was necessary for the python app indicator example to work.

In previous versions of Ubuntu, there was a package called indicator-applet to display the app indicator icon and menu on the desktop UI. The menu are coming from the application through the D-Bus. This package is part of the Unity graphical shell that Ubuntu was using. However since Ubuntu 17.10, Ubuntu switched to using GNOME as the graphical shell.  [Furthermore, the official design for GNOME are moving away from having status icon as part of the user interface](https://blogs.gnome.org/aday/2017/08/31/status-icons-and-gnome/). 

As a result, we need to install gnome-shell-extension-appindicator to continue to use the existing approach to create app indicator icon and menu. This is because the extension largely performs a similar functionality as indicator-applet described in the documentation.

### System overview

<figure>
<svg style="width: 100%; height: 300px;" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" width="1337px" height="451px" viewBox="-0.5 -0.5 1337 451" content="&lt;mxfile host=&quot;app.diagrams.net&quot; modified=&quot;2022-05-01T00:37:48.507Z&quot; agent=&quot;5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36&quot; etag=&quot;Vj6pLycnfZbzCCBE58Cb&quot; version=&quot;17.5.0&quot; type=&quot;google&quot; pages=&quot;3&quot;&gt;&lt;diagram id=&quot;bodo-_9RMdkC_etAQgBv&quot; name=&quot;Page-1&quot;&gt;5VfbctowEP0aHmF8wQYeuaRJaNKhgTRNXjLClm2lsuXKcsD9+q6wDDY2DJmEdjp9Ae9qLa/OObuSWuY4XF9yFAe3zMW0ZWjuumVOWoZh9PsD+JOeLPfoumnmHp8TV/l2jjn5hZVTU96UuDipBArGqCBx1emwKMKOqPgQ52xVDfMYrX41Rj6uOeYOonXvA3FFkHv7Rm/nv8LED4ov67ZacYiKYLWSJEAuW5Vc5kXLHHPGRP4UrseYSvQKXPL3Ph0Y3SbGcSROeaE95d40tKd3q/ThapYwfzactVWyr4imasGTZZqohEVWoJAIzn5sAYDUR4EIKTzq8AgLi2VcuPalCDoeZSsnQFx0XMKBk2cXCQRxq4AIPI+RI4NXEAo+j0VCkW50wUaU+BEYDiwKc3DUV6kW/oq5wOuSS636ErMQC55BiBpt64WYlAi7BSWrHaNmV/mCEpsD5UNKRP527h3O8KCgboad3OvfNPo8oimaITPBnkXnbasGex3yFQkpirAEWACYCiXtYzAxB1VITK0OidFrgMQ0z4WJXcMkzkTAIvmtOKbEQYJIy6aQymjJK3DZP1NZRxs9tZMNVEMI0O14vQGsGIcnX2whLMENuImqrGtSlOhCFnSoBkLiuvL1EcfwRbSkBT8xI5HYwGONWtZEzpUKlmeVV8ymnMaMMph3ErENyx6hdN+1qw7dbiT+qLpOVoNt7YmhpgW9qTyMc0mhV6MHu9CVlanA4SyNXOwqzBkHrfgsQvSGsVjh/IKFyBSAkoIqwQAiz77L9ztWYT6q6TbGZF2xMmWVWem+mZWEpdzBR1pyV+1yiPtYHIlT9SKROcoxxxQq57W6n72HsmNZVzpaXqkSrpNqtXugVqH6SeTK+ofaUJNCjvm8B8q5Ko6mnackhPcR+oYyk0IrF9qgqdAG9UKzP6DQjmnoHKzJc0SIo1ROgjmABAMolMhHyyTOA/9hKo09Kg3DauDSOg+X1zefH6/gyGI+9GaTJ2c66dl6W69xeYd9ksi9y9DmAok0+cIE8Qjm1wKHf7bFbo3Har890GI31gxzAtDIvffEvqt/cIsdnNhilSi0jmYOuhVZtIuj58ldWM0+k2eIUgjzvARy3RfJNomTdONmt+mi//Vee10sF0+Lp2HPeGnQzVjdpeRVS/4EGH4b7wX/kWbss2lG79rV45f59yQD5u5GmofvLvbmxW8=&lt;/diagram&gt;&lt;diagram id=&quot;o8q4UXIUv6cXhgm4nUvg&quot; name=&quot;Page-2&quot;&gt;7Vnbdto6EP0aHmH5hoHHQNI0Z+X0stLbyUuXsIWtVpZcWQ7Qr+/IljHCGEgKJD2rD4A1HsvS3ntGI9FxJ8niWqA0/peHmHYcKxIk7LiXHcex4QOGFEXYMCiPO/KzMlrampMQZ4aj5JxKkprGgDOGA2nYkBB8brrNOG0O4y5AFDesn0ko49I6dAa1/TUmUVy9yPZH5Z0EVc564FmMQj5fM7lXHXciOJflVbKYYKqQqXApn3vVcnc1MIGZPOSBD+6X+Md3j0zu59fvP3S/yoSTru7lAdFcT/hymmd6wHJZoZBJwb+vAIChj2OZULi04RImliq/ZBEpgnszyudBjITshUQABV9DJBH4zWMi8V2KAuU8B1ewzTiTmmPHgzaiJGLQCGBSWIBBjw8LiRetE7dXcILIME+wFEtwqR6wNAPLsu35/bI9rwl1Pe0Tr5E50jakJROtuq5hhguN9CNQdxqoNxGfk4QihhW+ErDUIFkmZrZ3HIi6vmdi5FaYrWHkbcPIc44A0mU0Rou39j/0Y4rlNfk0Ht9Pu24DExxCaOomFzLmEWeIXtXWseA5C3Gogap9bjlPtVq/YSmXGj6US25qGfASyy/q+V6/av6nuysalwujtdzJiRpwKyO7lJHxXAR4h59WMCgjwrv6G2ynXWCKJHkwB3d0Br2GzAOBkcRg49NvkBkg0Vh5yhn8MDyH7zuJZJ694ZLMCBY3Eid/RfD7IrCt7Sqogt/qDUZDvaAdrAzd3TtOYC61C5/NMhjMpnRWb326mvr7k+YZtPFSOT5XoO8apEGNT6Vau1LEDI78H7kqeAocu1kB5AU42F66KNCs7sNVpH7NjPAZySCGukD3DoMtX1D6NvRgsr2t/tjL7J5ltZ3b9nLEtXtO31ht/eZiaw8GvX5zufVPVZL4O4OJcfasmfXUUTY4MMpGzxllg9YoU9L9rSi7YSEJkOSiDLebQK3JPkpUiLBplpauVcyVr/uTYs51embErbaT6yHXP2/IDVv5DMnDQXT6QGeDy0eLoq0Xg/6Iway7WQzz6uKFxCwjnHVRmpJKOq0CWZmLaZnWo8x0Kva9bEOjoC65sYctNrgTTmEeq3Q3I5RumBr7U6VVmD+90DcSEoZFpbFN+WZsnErs/U2xu02tu+4WpXunUvrodJlLHVckmOXqxIcSXOju/5O3+o1SYXVQsZ64nC0b85OlrSpznoLNizS92ZtP/gTiHH9wyIrjOKdaccbW4Fryt/49mw69bv6zf/v61ZbTvluSQS7vqHNQ8fj99/HLQqMorGvElrKwaL3DggA8Khs/dRduH1gmHrzh3qkN2G9bI8cQR6WNl7v7bkpnog/Y1fm7+orxXwHtE1DF81MVdJ5zu+bxtOY6M8kOi78KrAyLB7UnP+/+0RquU22/MKJHz8szNOs/lsoUUP/35l79Ag==&lt;/diagram&gt;&lt;diagram id=&quot;ccWSbkW6FTuylw-CS0OS&quot; name=&quot;Page-3&quot;&gt;1Vldd9o4EP01PMLB3+aRQNKy3d0my8lp96lH2MJWV1iuLAL013dky9jGxpAFk/QJayyk0b1zRyO5Z0xW2w8cxeFfzMe0pw/9bc+Y9nRdd90R/EjLLrNoQ0fPLAEnvrIVhjn5ifOOyromPk4qHQVjVJC4avRYFGFPVGyIc7apdlsyWp01RgGuGeYeonXrF+KLMLO6ulPYP2IShGpme6j8XqG8rzIkIfLZpmQy7nvGhDMmsqfVdoKpBC+HJfvfw5G3e784jsQ5f3iej5zxbvbpj6flwzOJn42n4GtfjfKC6Fqtd7pYJ8phsctBSARn/+3XD67fhWJF4VGDR1hYLPuttoGMgcGSso0XIi4GPuFAyTcfCQT9NiEReB4jT3beQFewLVkkFOe6CW1ESRBBw4NFYQ6G+ipzlzEXeFsyqVV/wGyFBd9BF/W2rw3NgWNlf1NhCDxl7U3BqWEqW1jic6RsSIVRsB++gBoeFNrNyC8+P7K+s/hn9vKDeMvPrrD8nznyJZCxD4GnmoyLkAUsQvS+sN5xto58LEcdQqvo8ydjsSLjOxZipxBFa8EOqBJAy1iqQmJMUZIQLzc/EJp3y1yT/hyNtdx9tuYebgkwJXaYIMDiVCDWKeSYIkFeqn5cwkWbkyUV1BWwISuKIpyjpSAeXidEDdsa6NUI1cx6hOpOQ4QaxhVCtBEWowZLvIOQi+RccUyJB9TIlk3BlbsFryBm/1jLzJYqvJ+kaI2hg27G2xSz/D08BWKPYglxgE5Uo7eWHCTA4AUdqxcr4vuZUDDMiBY0pyhmJBIpPNZdz5rKsUAbmVdKGDLBTRhlMO40YinRS1DEoekgX9W5bw2w8wNiWE1XVj0WtKZY6CoUzNZspcC5OD0BiHz3Vf5/YOXNf9VwaWO6rbR2qlViRXs9KydzmHVmDrMb8+ZbJTWrIallSpVwnaXV4RGtgvpJ5Ev9gzbUoOBjNu4ROVeDo6kWKAVCWWbNKfY6MrNloJWFNqonXc0c1YVmdyU0uzvWZGW3wtFaDoI5gAQv0EoiHy2SOOv4+1Jp2oc76D4blsl0nJzxa9PZWOXVC4kbVHlFGq0k0SKnHkmjaesRcwJrl/trauy0ZHxC3+4+Th5H9xqNH4Jl8olN7LwyfjclY5uT/7NkPL1hvf6UY1vO4SnHrQvA1I16+JvmaGC4HSFlnkbq4og/jWcbiScjsqEAaOznvFUB0Ly69iPm+y/absJZXka/E9KOV21JjKKL9v+5QGKd/M0EWRLMvyDhhbhcvmUT3GbPb82pZ+c8Tasek5yGI7Pm7rPiTXb89pPSO9vxD4mqbPYqQZR3+oLM1p3+IuE6Zwp39JYydS4u0zX3iExn+ckq0+vMS69Yblqoa26XonUOzlz7m4xyne52Vqc3rsI9yqdPXi6813rl8a1loEoQBBGsvZ+EsLo+3gocJYRF/bOO5ntzuriq9Vrrvfg6sNnJk5eE597knX+Z2KSbqrJO3wpeRzp9Uz885Dacca1ho3aucTfYWkF1cWcxjuPZG980XYc6YOV00mv6ANVZyht1x1rppsmjBKdc3f6mKfv6KL9QGFOoOiTK3W1qZk2YDd9vtJFZJ9h6PcHQLL4bp+9KX9+N+18=&lt;/diagram&gt;&lt;/mxfile&gt;"><defs/><g><path d="M 626.47 180 L 904.03 180 C 921.27 180 935.25 200.15 935.25 225 C 935.25 249.85 921.27 270 904.03 270 L 626.47 270 C 609.23 270 595.25 249.85 595.25 225 C 595.25 200.15 609.23 180 626.47 180 Z" fill="rgb(255, 255, 255)" stroke="rgb(0, 0, 0)" stroke-width="2" stroke-miterlimit="10" pointer-events="all"/><path d="M 904.03 180 C 886.78 180 872.8 200.15 872.8 225 C 872.8 249.85 886.78 270 904.03 270" fill="none" stroke="rgb(0, 0, 0)" stroke-width="2" stroke-miterlimit="10" pointer-events="all"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility" style="overflow: visible; text-align: left;"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 338px; height: 1px; padding-top: 225px; margin-left: 596px;"><div data-drawio-colors="color: rgb(0, 0, 0); " style="box-sizing: border-box; font-size: 0px; text-align: center;"><div style="display: inline-block; font-size: 24px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; pointer-events: all; white-space: normal; overflow-wrap: normal;">D-Bus</div></div></div></foreignObject><text x="765" y="232" fill="rgb(0, 0, 0)" font-family="Helvetica" font-size="24px" text-anchor="middle">Dbus</text></switch></g><path d="M 1058.88 225 L 941.62 225" fill="none" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="stroke"/><path d="M 1064.13 225 L 1057.13 228.5 L 1058.88 225 L 1057.13 221.5 Z" fill="rgb(0, 0, 0)" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="all"/><path d="M 936.37 225 L 943.37 221.5 L 941.62 225 L 943.37 228.5 Z" fill="rgb(0, 0, 0)" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="all"/><path d="M 1065.25 60 L 1065.25 60 L 1335.25 60 L 1335.25 60" fill="rgb(255, 255, 255)" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="all"/><path d="M 1065.25 60 L 1065.25 390 L 1335.25 390 L 1335.25 60" fill="none" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="none"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility" style="overflow: visible; text-align: left;"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 1px; height: 1px; padding-top: 100px; margin-left: 1200px;"><div data-drawio-colors="color: rgb(0, 0, 0); " style="box-sizing: border-box; font-size: 0px; text-align: center;"><div style="display: inline-block; font-size: 24px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; pointer-events: none; white-space: nowrap;">python application<br style="font-size: 24px" /></div></div></div></foreignObject><text x="1200" y="107" fill="rgb(0, 0, 0)" font-family="Helvetica" font-size="24px" text-anchor="middle">python application&#xa;</text></switch></g><path d="M 1200.25 210 L 1200.25 283.63" fill="none" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="none"/><path d="M 1200.25 288.88 L 1196.75 281.88 L 1200.25 283.63 L 1203.75 281.88 Z" fill="rgb(0, 0, 0)" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="none"/><rect x="1125.75" y="150" width="149" height="60" fill="rgb(255, 255, 255)" stroke="rgb(0, 0, 0)" pointer-events="none"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility" style="overflow: visible; text-align: left;"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 147px; height: 1px; padding-top: 180px; margin-left: 1127px;"><div data-drawio-colors="color: rgb(0, 0, 0); " style="box-sizing: border-box; font-size: 0px; text-align: center;"><div style="display: inline-block; font-size: 20px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; pointer-events: none; white-space: normal; overflow-wrap: normal;"><font style="font-size: 20px">appindicator</font></div></div></div></foreignObject><text x="1200" y="186" fill="rgb(0, 0, 0)" font-family="Helvetica" font-size="20px" text-anchor="middle">appindicator</text></switch></g><rect x="1111.5" y="290" width="177.5" height="60" fill="rgb(255, 255, 255)" stroke="rgb(0, 0, 0)" pointer-events="none"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility" style="overflow: visible; text-align: left;"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 176px; height: 1px; padding-top: 320px; margin-left: 1113px;"><div data-drawio-colors="color: rgb(0, 0, 0); " style="box-sizing: border-box; font-size: 0px; text-align: center;"><div style="display: inline-block; font-size: 20px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; pointer-events: none; white-space: normal; overflow-wrap: normal;"><font style="font-size: 20px">Dbusmenu server </font></div></div></div></foreignObject><text x="1200" y="326" fill="rgb(0, 0, 0)" font-family="Helvetica" font-size="20px" text-anchor="middle">Dbusmenu server </text></switch></g><path d="M 471.62 224.7 L 530.3 224.7 L 588.88 224.97" fill="none" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="none"/><path d="M 466.37 224.7 L 473.37 221.2 L 471.62 224.7 L 473.37 228.2 Z" fill="rgb(0, 0, 0)" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="none"/><path d="M 594.13 224.99 L 587.12 228.46 L 588.88 224.97 L 587.15 221.46 Z" fill="rgb(0, 0, 0)" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="none"/><path d="M 42.25 0 L 42.25 0 L 465.25 0 L 465.25 0" fill="rgb(255, 255, 255)" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="none"/><path d="M 42.25 0 L 42.25 449.38 L 465.25 449.38 L 465.25 0" fill="none" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="none"/><path d="M 238.46 130 L 180.56 204.96" fill="none" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="none"/><path d="M 177.35 209.12 L 178.86 201.44 L 180.56 204.96 L 184.4 205.71 Z" fill="rgb(0, 0, 0)" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="none"/><path d="M 289.07 130 L 357.95 205.3" fill="none" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="none"/><path d="M 361.5 209.18 L 354.19 206.37 L 357.95 205.3 L 359.35 201.65 Z" fill="rgb(0, 0, 0)" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="none"/><rect x="152.25" y="70" width="218.75" height="60" fill="rgb(255, 255, 255)" stroke="rgb(0, 0, 0)" pointer-events="none"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility" style="overflow: visible; text-align: left;"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 217px; height: 1px; padding-top: 100px; margin-left: 153px;"><div data-drawio-colors="color: rgb(0, 0, 0); " style="box-sizing: border-box; font-size: 0px; text-align: center;"><div style="display: inline-block; font-size: 20px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; pointer-events: none; white-space: normal; overflow-wrap: normal;"><span style="font-size: 20px">StatusNotifierWatcher</span></div></div></div></foreignObject><text x="262" y="106" fill="rgb(0, 0, 0)" font-family="Helvetica" font-size="20px" text-anchor="middle">StatusNotifierWatcher</text></switch></g><path d="M 153.6 270 L 153.6 305 L 153.52 333.63" fill="none" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="none"/><path d="M 153.5 338.88 L 150.02 331.87 L 153.52 333.63 L 157.02 331.89 Z" fill="rgb(0, 0, 0)" stroke="rgb(0, 0, 0)" stroke-miterlimit="10" pointer-events="none"/><rect x="59.75" y="210" width="187.5" height="60" fill="rgb(255, 255, 255)" stroke="rgb(0, 0, 0)" pointer-events="none"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility" style="overflow: visible; text-align: left;"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 186px; height: 1px; padding-top: 240px; margin-left: 61px;"><div data-drawio-colors="color: rgb(0, 0, 0); " style="box-sizing: border-box; font-size: 0px; text-align: center;"><div style="display: inline-block; font-size: 18px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; pointer-events: none; white-space: normal; overflow-wrap: normal;"><font style="font-size: 18px">IndicatorStatusIcon </font></div></div></div></foreignObject><text x="154" y="245" fill="rgb(0, 0, 0)" font-family="Helvetica" font-size="18px" text-anchor="middle">IndicatorStatusIcon </text></switch></g><g transform="translate(-0.5 -0.5)"><switch><foreignObject pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility" style="overflow: visible; text-align: left;"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 506px; height: 1px; padding-top: 45px; margin-left: 1px;"><div data-drawio-colors="color: rgb(0, 0, 0); " style="box-sizing: border-box; font-size: 0px; text-align: center;"><div style="display: inline-block; font-size: 24px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; pointer-events: none; white-space: normal; overflow-wrap: normal;"><div style="font-size: 24px"><font style="font-size: 24px"> gnome-shell-extension-appindicator</font></div><div style="font-size: 24px"><br style="font-size: 24px" /></div></div></div></div></foreignObject><text x="254" y="52" fill="rgb(0, 0, 0)" font-family="Helvetica" font-size="24px" text-anchor="middle"> gnome-shell-extension-appindicator&#xa;</text></switch></g><rect x="292.25" y="210" width="140" height="60" fill="rgb(255, 255, 255)" stroke="rgb(0, 0, 0)" pointer-events="none"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility" style="overflow: visible; text-align: left;"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 138px; height: 1px; padding-top: 240px; margin-left: 293px;"><div data-drawio-colors="color: rgb(0, 0, 0); " style="box-sizing: border-box; font-size: 0px; text-align: center;"><div style="display: inline-block; font-size: 20px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; pointer-events: none; white-space: normal; overflow-wrap: normal;"><font style="font-size: 20px">AppIndicator</font></div></div></div></foreignObject><text x="362" y="246" fill="rgb(0, 0, 0)" font-family="Helvetica" font-size="20px" text-anchor="middle">AppIndicator</text></switch></g><rect x="128.5" y="268" width="50" height="194" fill="rgb(255, 255, 255)" stroke="rgb(0, 0, 0)" transform="rotate(90,153.5,365)" pointer-events="none"/><g transform="translate(-0.5 -0.5)"><switch><foreignObject pointer-events="none" width="100%" height="100%" requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility" style="overflow: visible; text-align: left;"><div xmlns="http://www.w3.org/1999/xhtml" style="display: flex; align-items: unsafe center; justify-content: unsafe center; width: 192px; height: 1px; padding-top: 365px; margin-left: 58px;"><div data-drawio-colors="color: rgb(0, 0, 0); " style="box-sizing: border-box; font-size: 0px; text-align: center;"><div style="display: inline-block; font-size: 20px; font-family: Helvetica; color: rgb(0, 0, 0); line-height: 1.2; pointer-events: none; white-space: normal; overflow-wrap: normal;"><font style="font-size: 20px">Dbusmenu client </font></div></div></div></foreignObject><text x="154" y="371" fill="rgb(0, 0, 0)" font-family="Helvetica" font-size="20px" text-anchor="middle">Dbusmenu client </text></switch></g></g><switch><g requiredFeatures="http://www.w3.org/TR/SVG11/feature#Extensibility"/><a transform="translate(0,-5)" xlink:href="https://www.diagrams.net/doc/faq/svg-export-text-problems" target="_blank"><text text-anchor="middle" font-size="10px" x="50%" y="100%">Text is not SVG - cannot display</text></a></switch></svg>
<figcaption style="text-align: center; font-style: italic;">Diagram of the related components</figcaption>
</figure>

When the python application starts, it creates an app indicator object with information such as the icon file path and the type of status icon. Then the application creates a `GtkMenu`  object and pass the menu to the app indicator. The app indicator will pass the menu to dbusmenu library and parse the `GtkMenu` object recursively to create `DbusmenuMenu` and `DbusmenuMenuItem`. After that, the app indicator will create a dbusmenu server and register a StatusNotifierItem on the D-Bus. 

The D-Bus is a message bus system that is available on Ubuntu for communication between processes and applications. There is a daemon running in the background to route messages. The python application uses the D-Bus to communicate with the extension and vice versa. For further information and general concepts about D-Bus, refer to [the officiial tutorial](https://dbus.freedesktop.org/doc/dbus-tutorial.html). To check the D-Bus messages exchanged for app indicator icon, we can watch the D-Bus messages using `dbus-monitor` command in the terminal. 
 
For example, the following message was emitted to the daemon when the python application registers status notifier item on the D-Bus.


```console
method call time=1652169917.112493 sender=:1.184 -> destination=:1.37 serial=14 path=/StatusNotifierWatcher; interface=org.kde.StatusNotifierWatcher; member=RegisterStatusNotifierItem
   string "/org/ayatana/NotificationItem/example_simple_client"
```

- The `sender` is 1.184, representing the python application.
- The `destination` is 1.37, representing the extension.
- The `path` is the object path where the D-Bus message is send to.
- The `interface` is the public contract of the object on the path, indicating the members we can call on the object path.
- The `member` is the member of the object that we are calling through this message. It could be a method that's return value or a method to emit a signal. In this case, we are calling `RegisterStatusNotifierItem` which is used to emit a signal to register the status notifier item.
- The string value is the notification item's path that we pass in when calling `RegisterStatusNotifierItem`


Next, gnome-shell-extension-appindicator have a watcher running in the background, watching for any new StatusNotiferItem on the D-Bus. Upon new StatusNotifierItem, the watcher will create a proxy app indicator object and connect the proxy app indicator to the StatusNotifierItem. The proxy app indicator is used to access the app indicator information from the StatusNotifierItem.


_Truncated message for retrieving information on the StatusNotifierItem from the extension by calling `GetAll`_
```console
method call time=1652169917.115926 sender=:1.37 -> destination=:1.184 serial=1354 path=/org/ayatana/NotificationItem/example_simple_client; interface=org.freedesktop.DBus.Properties; member=GetAll
   string "org.kde.StatusNotifierItem"
method return time=1652169917.116081 sender=:1.184 -> destination=:1.37 serial=15 reply_serial=1354
   array [
      dict entry(
         string "Id"
         variant             string "example-simple-client"
      )
      dict entry(
         string "Category"
         variant             string "ApplicationStatus"
      )
      dict entry(
         string "Status"
         variant             string "Active"
      )
      dict entry(
         string "IconName"
         variant             string "/home/wal8800/workspace/github-notifier/images/mushroom.png"
      )
      dict entry(
         string "Menu"
         variant             object path "/org/ayatana/NotificationItem/example_simple_client/Menu"
      )
      ...
   ]
```


The watcher also creates an indicator status icon object with a dbusmenu client. The indicator status icon object represents the app indicator icon and menu that is going to be render on the UI. The dbusmenu client connects to the dbusmenu server using the menu path from the StatusNotifierItem. Once the indicator status icon object is created, the extension will render the app indicator icon and menu on the Desktop UI.

Any state changes or events on app indicator menu from the python application will sync across to the extension and vice versa. This occurs in the communication betweeen the dbusmenu server and dbusmenu client. For example, when I open up the menu by clicking on the icon then clicking on one of the items and closing menu. There are messages for each of the events.


_sending "opened" event to the python application from the extension and triggering "AboutToShow" hooks_
```console
method call time=1652170713.678815 sender=:1.37 -> destination=:1.184 serial=1446 path=/org/ayatana/NotificationItem/example_simple_client/Menu; interface=com.canonical.dbusmenu; member=Event
   int32 0
   string "opened"
   variant       int32 0
   uint32 0

method call time=1652170715.908974 sender=:1.37 -> destination=:1.184 serial=1446 path=/org/ayatana/NotificationItem/example_simple_client/Menu; interface=com.canonical.dbusmenu; member=AboutToShow
   int32 0
method return time=1652170715.909142 sender=:1.184 -> destination=:1.37 serial=23 reply_serial=1445
method return time=1652170715.909215 sender=:1.184 -> destination=:1.37 serial=24 reply_serial=1446
   boolean false
```

_sending "clicked" event_
```console
method call time=1652170717.274419 sender=:1.37 -> destination=:1.184 serial=1447 path=/org/ayatana/NotificationItem/example_simple_client/Menu; interface=com.canonical.dbusmenu; member=Event
   int32 5
   string "clicked"
   variant       int32 0
   uint32 0
method return time=1652170717.274636 sender=:1.184 -> destination=:1.37 serial=25 reply_serial=1447
```

_sending "closed" event_
```console
method call time=1652170717.275973 sender=:1.37 -> destination=:1.184 serial=1448 path=/org/ayatana/NotificationItem/example_simple_client/Menu; interface=com.canonical.dbusmenu; member=Event
   int32 0
   string "closed"
   variant       int32 0
   uint32 0
```

Finally, when the python application exits, the StatusNotifierItem is deregistered. The extension's watcher will pick up the deregistration and clean up the proxy app indicator along with the indicator status icon.

### So can we style the app indicator's menu item?  

Diving into the dbusmenu library, the python application's dbusmenu server doesn't pass markup styled text to the indicator status icon's dbusmenu client in the extension. The server sync across:

- The menu structure 
- The menu content (plain label string)
- The menu action/triggers

This means if the menu's label that we are passing to the extensions contains markup style, the styling doesn't get passed to the app indicator menu's label. Hence why we didn't see the bold styling in the modified example.

## What's next

To make the markup styling appear on the app indicator menu, one approach is to modify the extension and the dbusmenu library to allow styling to be passed through. However, that requires changing multiple components and maintaining these modified libraries.

Another option is to use [gjs](https://gjs.guide/) to build out the app indicator application from scratch. It is a fully featured javascript sdk for building GNOME desktops GUI application and have amazing support for styling the UI. Furthermore, the extension is already built with gjs, this means we can build our own custom app indicator icon application.

Lastly, we can also use the non-styled app indicator menu to open up a window that allows us to fully customise the style and the content within the window. For example, jetbrain's toolbox application have this behaviour.

Overall, it looks like there is a decent amount of work to get an equivalent application on Ubuntu so I won't try to create my own app indicator application. Currently, I find using github slack notification and the github pull requests page is sufficient.
