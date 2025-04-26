pragma solidity >= 0.8.11 <= 0.8.11;
pragma experimental ABIEncoderV2;
//FoodSupply solidity code
contract FoodSupply {

    uint public productCount = 0; 
    mapping(uint => product) public productList; 
     struct product
     {
       string user;
       string productid;
       string productname;
       string desc;
       string image;
       string date;
       string tracing_type;
       string location;
       string status;
     }
 
   // events 
   event productCreated(uint indexed _productId);

  
   //function  to save product details
   function createProduct(string memory usr, string memory pid, string memory pname, string memory de, string memory img, string memory dd, string memory ttype, string memory loc, string memory status) public {
      productList[productCount] = product(usr, pid, pname, de, img, dd, ttype, loc, status);
      emit productCreated(productCount);
      productCount++;
    }

     //get product count
    function getProductCount()  public view returns (uint) {
          return  productCount;
    }

    function getLocation(uint i) public view returns (string memory) {
        product memory chq = productList[i];
	return chq.location;
    }

    function getOwner(uint i) public view returns (string memory) {
        product memory chq = productList[i];
	return chq.user;
    }

    function getProductid(uint i) public view returns (string memory) {
        product memory chq = productList[i];
	return chq.productid;
    }

    function getProductname(uint i) public view returns (string memory) {
        product memory chq = productList[i];
	return chq.productname;
    }

    function getDesc(uint i) public view returns (string memory) {
        product memory chq = productList[i];
	return chq.desc;
    }

    function getImage(uint i) public view returns (string memory) {
        product memory chq = productList[i];
	return chq.image;
    }

    function getDate(uint i) public view returns (string memory) {
        product memory chq = productList[i];
	return chq.date;
    }

     function getTracing(uint i) public view returns (string memory) {
        product memory chq = productList[i];
	return chq.tracing_type;
    }

     function getStatus(uint i) public view returns (string memory) {
        product memory chq = productList[i];
	return chq.status;
    }

     
       
    uint public userCount = 0; 
    mapping(uint => user) public usersList; 
     struct user
     {
       string username;
       string password;
       string phone;
       string usertype;
     }
 
   // events
 
   event userCreated(uint indexed _userId);
 
  function createUser(string memory _username, string memory _password, string memory _phone, string memory _utype) public {
      usersList[userCount] = user(_username, _password, _phone, _utype);
      emit userCreated(userCount);
      userCount++;
    }

    
     //get user count
    function getUserCount()  public view returns (uint) {
          return  userCount;
    }

    function getUsername(uint i) public view returns (string memory) {
        user memory usr = usersList[i];
	return usr.username;
    }

    function getPassword(uint i) public view returns (string memory) {
        user memory usr = usersList[i];
	return usr.password;
    }

    function getUserType(uint i) public view returns (string memory) {
        user memory usr = usersList[i];
	return usr.usertype;
    }
}