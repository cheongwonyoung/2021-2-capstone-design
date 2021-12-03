'use strict'

// navbar
const navbar = document.querySelector('#navbar');
const navbarHeight = navbar.getBoundingClientRect().height;
document.addEventListener('scroll', () => {
  if(window.scrollY > navbarHeight) {
    navbar.classList.add('navbar--dark');
  } else {
    navbar.classList.remove('navbar--dark');
  }
});

// navbar 메뉴 클릭할 때 스크롤 처리
const navbarMenu = document.querySelector('.navbar__menu');
navbarMenu.addEventListener('click', (event) => {
  const target = event.target;
  const link = target.dataset.link;
  if (link == null) {
    return;
  }
  navbarMenu.classList.remove('open');
  scrollIntoView(link);
});

// 작은 화면에서의 navbar toggle 버튼
const navbarToggleBtn = document.querySelector('.navbar__toggle-btn');
navbarToggleBtn.addEventListener('click', () => {
  navbarMenu.classList.toggle('open');
});

const homeContactBtn = document.querySelector('.home__contact');
homeContactBtn.addEventListener('click', () => {
  scrollIntoView('#footer');
});

// 스크롤 내릴 시 fadeout 처리
const home = document.querySelector('.home__container');
const homeHeight = home.getBoundingClientRect().height;
document.addEventListener('scroll', () => {
  home.style.opacity = 1 - window.scrollY / homeHeight;
});

// 아래로 스크롤 할 때 "arrow up" 버튼 표시
const arrowUp = document.querySelector('.arrow-up');
document.addEventListener('scroll', ()=> {
  if(window.scrollY > homeHeight / 2) {
    arrowUp.classList.add('visible');
  } else {
    arrowUp.classList.remove('visible');
  }
});

// "arrow up" 버튼 클릭 처리
arrowUp.addEventListener('click', ()=> {
  scrollIntoView('#home');
});

function scrollIntoView(selector) {
  const scrollTo = document.querySelector(selector);
  scrollTo.scrollIntoView({behavior: 'smooth'});
}


function showImage(e) {
  // e.preventDefault();

  // var image_file = document.getElementById('chooseFile').files[0];

  // var url = "/";
  // var request = new XMLHttpRequest();
  // request.open('POST', url, true);

  // var form_data = new FormData();
  // form_data.append('file', image_file);
  // request.setRequestHeader('Content-Type', 'multipart/form-data'); // set headers 
  // request.send(form_data);
  
  // // recv response message 
  // request.onreadystatechange = () => {
  //   if (request.readyState != 4) return; // not readystate
  //   if (request.status != 200) alert('Status : ' + request.status); // request failed
  //   else {      alert(request.responseText);    } // request successed 
  // }

  let newImage = document.getElementById('image-show').lastElementChild;
  newImage.style.display = "block";
  // let final_download = document.querySelector('.final_download');
  // final_download.style.visibility = "visible";

  document.getElementById('image-upload').style.visibility = 'hidden';
  document.getElementById('fileName').textContent = null; //기존 파일 이름 지우기
}

const submit = document.getElementById('submitButton');
submit.onclick = showImage;     //Submit 버튼 클릭시 이미지 보여주기

const btnUpload = document.querySelector('.btn-upload');
const inputFile = document.querySelector('input[type="file"]');
const uploadBox = document.querySelector('#service');

// 박스 안에 Drag 들어왔을 때
uploadBox.addEventListener('dragenter', function(e) {
  console.log('dragenter');
});

// 박스 안에 Drag 하고 있을 때
uploadBox.addEventListener('dragover', function(e) {
  e.preventDefault();
  console.log('dragover');

  this.style.backgroundColor = "green";
});

// 박스 밖으로 Drag가 나갈 때
uploadBox.addEventListener('dragleave', function(e) {
  console.log('dragleave');

  this.style.backgroundColor = "white";
})

// 박스 안에서 Drag를 Drop 했을 때
uploadBox.addEventListener('drop', function(e) {
  e.preventDefault();

  console.log('drop');
  this.style.backgroundColor = 'white';

  console.dir(e.dataTransfer);

  let data = e.dataTransfer.files[0];
  console.dir(data);

  let name = document.getElementById('fileName');
  name.textContent = data.name;

  let newImage = document.createElement("img");
  newImage.setAttribute("class", 'img');

  newImage.src = URL.createObjectURL(data);   
  newImage.style.display = "block";

  let container = document.getElementById('image-show');
  container.appendChild(newImage);  
});

// local 에서 파일 input 되었을 때
function loadFile(input) {
  let file = input.files[0];

  let name = document.getElementById('fileName');
  name.textContent = file.name;

  let newImage = document.createElement("img");
  newImage.setAttribute("class", 'img');

  newImage.src = URL.createObjectURL(file);   

  newImage.style.display = "block";

  let container = document.getElementById('image-show');
  container.appendChild(newImage);
};

