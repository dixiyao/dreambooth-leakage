"""
A tool for notifying iOS devices of starting and completing the programs.
"""

import http.client
import urllib


def send_notification(msg):
    conn = http.client.HTTPSConnection("api.pushover.net:443")
    conn.request(
        "POST",
        "/1/messages.json",
        urllib.parse.urlencode(
            {
                "token": "ah6q72iauvz2b6jvpvskh2too5boe9",
                "user": "ud51hrjiaev842zfr3knab9wc6x7p5",
                "message": msg,
            }
        ),
        {"Content-type": "application/x-www-form-urlencoded"},
    )
    conn.getresponse()


if __name__ == "__main__":
    send_notification("The program is completed.")
    print("Notification sent.")
