import io
from pathlib import Path
from slackclient import SlackClient


def send_message(client, text, channel):
    """Send message to slack.

    Args:
        client (SlackClient): slack client instance.
        text (str): message to send.
        channel (str): target slack channel to send message.

    Returns:
        A boolean object which sending message succeed or not.
    """
    result = client.api_call(
        "chat.postMessage",
        channel=channel,
        text=text
    )
    return 'ok' in result


class SlackOut(io.TextIOWrapper):
    """Output stream to slack.
    """
    def __init__(self, channel='bot', tokenfile='slack_token'):
        token = Path(tokenfile).read_text()
        self._client = SlackClient(token)
        self._channel = channel

    def write(self, text):
        return send_message(self._client, text, self._channel)
