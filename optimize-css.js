/*

Even though we're using nodejs and this js file, the site is still
technically 100% JS-free because this is just used to build the site.

*/

const critical = require('critical');
const fs = require('fs');
const path = require('path');

const distDir = '_site';

// Function to find all HTML files in all subdirectories
const getHtmlFiles = (dir, fileList = []) => {
  const files = fs.readdirSync(dir);
  files.forEach((file) => {
    const filePath = path.join(dir, file);
    if (fs.statSync(filePath).isDirectory()) {
      getHtmlFiles(filePath, fileList);
    } else if (file.endsWith('.html')) {
      // We need the path relative to _site for the critical tool
      fileList.push(path.relative(distDir, filePath));
    }
  });
  return fileList;
};

async function optimize() {
  const files = getHtmlFiles(distDir);

  for (const file of files) {
    console.log(`Optimizing: ${file}`);
    await critical.generate({
      inline: true,
      base: distDir,
      src: file,
      target: file, 
      width: 1300,
      height: 900,
      // Critical automatically "purges" unused CSS for the specific HTML file
    });
  }
}

optimize().catch(err => {
  console.error(err);
  process.exit(1);
});