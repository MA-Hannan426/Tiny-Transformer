"""
Interactive command-line chat interface for TinyTransformerLM.
"""

from inference.predictor import Predictor


class ChatSession:
    """
    Interactive chat session with TinyTransformerLM.
    """

    def __init__(
        self,
        checkpoint_path,
    ):

        self.predictor = Predictor(
            checkpoint_path=checkpoint_path
        )

    ##########################################################
    # Banner
    ##########################################################

    def display_banner(self):

        print("\n" + "=" * 60)
        print("        TinyTransformerLM Interactive Chat")
        print("=" * 60)
        print("Type 'exit' or 'quit' to end the session.\n")

    ##########################################################
    # Handle User Input
    ##########################################################

    def process_prompt(
        self,
        prompt,
    ):

        response = self.predictor.predict(
            prompt
        )

        return response

    ##########################################################
    # Chat Loop
    ##########################################################

    def start(self):

        self.display_banner()

        while True:

            prompt = input("You : ").strip()

            if prompt.lower() in (
                "exit",
                "quit",
            ):

                print("\nGoodbye!\n")
                break

            if prompt == "":

                continue

            response = self.process_prompt(
                prompt
            )

            print("\nModel :")
            print(response)
            print("\n" + "-" * 60 + "\n")


##############################################################
# Entry Point
##############################################################

def main():

    checkpoint_path = (
        "checkpoints/best_model.pt"
    )

    chat = ChatSession(
        checkpoint_path
    )

    chat.start()


if __name__ == "__main__":

    main()