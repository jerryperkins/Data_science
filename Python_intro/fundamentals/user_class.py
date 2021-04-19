class User:
    def __init__(self, username, email_address):
        self.name = username
        self.email = email_address
        self.account_balance = 0
    
    def make_deposit(self, amount):
        self.account_balance += amount
    def make_withdrawal(self, amount):
        self.account_balance = self.account_balance - amount
    def display_user_balance(self):
        print(self.account_balance)
    def transfer_money(self, other_user, amount):
        self.account_balance = self.account_balance - amount
        other_user.account_balance += amount

    



paul = User("Paul", "paul@sdojo.com")
emily = User("Emily", "emily@dojo.com")
sam = User("Sam", "sam@dojo.com")

paul.make_deposit(100)
paul.make_deposit(10)
paul.make_deposit(50)
paul.make_withdrawal(80)
paul.display_user_balance()

emily.make_deposit(75)
emily.make_deposit(35)
emily.make_withdrawal(55)
emily.make_withdrawal(25)
emily.display_user_balance()

sam.make_deposit(350)
sam.make_withdrawal(75)
sam.make_withdrawal(100)
sam.make_withdrawal(125)
sam.display_user_balance()

paul.transfer_money(sam, 50)
paul.display_user_balance()
sam.display_user_balance()


# print(paul.name)
# print(paul.email)
# print(paul.account_balance)
# print(emily.name)
# paul.make_deposit(100)
# print(paul.account_balance)

